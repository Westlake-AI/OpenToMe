# /opentome/models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, Block as TimmBlock
from timm.layers import trunc_normal_
from timm.models.registry import register_model

# from opentome.timm.tome import tome_apply_patch
from opentome.timm.dtem import DTEMBlock
from opentome.tome.tome import token_unmerge_from_map, parse_r
from opentome.timm.bias_local_attn import LocalBlock
from opentome.utils.thetopk import ThreTopK


MERGENET_SMALL_CANONICAL_KWARGS = {
    "lambda_local": 4.0,
    "total_merge_latent": 0,
    "dtem_window_size": 8,
    "use_softkmax": True,
    "swa_size": 256,
    "dtem_feat_dim": 64,
    "dtem_t": 1,
}


def _with_mergenet_small_defaults(kwargs):
    """Canonical single-branch MergeNet-B config."""
    resolved = dict(kwargs)
    for key, value in MERGENET_SMALL_CANONICAL_KWARGS.items():
        if resolved.get(key, None) is None:
            resolved[key] = value
    return resolved


def _with_mergenet_branch_b_defaults(common_kwargs):
    """Branch B is the same compressed MergeNet as ``mergenet_small_cls``."""
    resolved = _with_mergenet_small_defaults(common_kwargs)
    resolved["lambda_local"] = MERGENET_SMALL_CANONICAL_KWARGS["lambda_local"]
    resolved["total_merge_latent"] = MERGENET_SMALL_CANONICAL_KWARGS["total_merge_latent"]
    resolved["use_softkmax"] = MERGENET_SMALL_CANONICAL_KWARGS["use_softkmax"]
    return resolved


def _torch_load_checkpoint(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class MyCrossAttention(nn.Module):
    """
    Implements multi-head cross attention where query and key/value sequences
    can have different lengths and batch sizes.
    
    Args:
        embed_dim: int, embedding dimension of input features
        num_heads: int, number of attention heads
        bias: bool, if True, add bias to qkv projections
        attn_drop: float, dropout rate for attention weights
        proj_drop: float, dropout rate after projection
    """

    def __init__(self, embed_dim, num_heads=8, bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Pre-norm for cross-attention stability.
        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)

        # q from seq_q, k/v from seq_kv
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, mask=None):
        """
        Args:
            q: (Bq, Nq, C)      -- queries
            kv: (Bk, Nk, C)     -- keys/values
            mask: (Bq, Nq, Nk), optional -- attention bias (additive)
        Returns:
            context: (Bq, Nq, C)
        """
        orig_dtype = q.dtype
        # Pre-norm before projections to suppress activation drift.
        q = self.q_norm(q)
        kv = self.kv_norm(kv)

        Bq, Nq, C = q.shape
        Bk, Nk, Ck = kv.shape
        assert C == self.embed_dim and Ck == self.embed_dim

        # Local fp32 guard: keep cross-attention numerics stable under AMP.
        # We intentionally run q/k/v and out_proj in fp32, then cast back.
        if q.is_cuda:
            with torch.amp.autocast(device_type='cuda', enabled=False):
                q_proj = self.q_proj(q.float())   # (Bq, Nq, C)
                k_proj = self.k_proj(kv.float())  # (Bk, Nk, C)
                v_proj = self.v_proj(kv.float())  # (Bk, Nk, C)
        else:
            print("error")
            q_proj = self.q_proj(q.float())   # (Bq, Nq, C)
            k_proj = self.k_proj(kv.float())  # (Bk, Nk, C)
            v_proj = self.v_proj(kv.float())  # (Bk, Nk, C)

        # Reshape for multi-head: (B, N, C) -> (B, num_heads, N, head_dim)
        q_proj = q_proj.reshape(Bq, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = k_proj.reshape(Bk, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v_proj = v_proj.reshape(Bk, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle broadcasting in batch dimension
        if Bq != Bk:
            if Bq == 1:
                # broadcast q over Bk
                q_proj = q_proj.expand(Bk, -1, -1, -1)
                B = Bk
            elif Bk == 1:
                # broadcast k/v over Bq
                k_proj = k_proj.expand(Bq, -1, -1, -1)
                v_proj = v_proj.expand(Bq, -1, -1, -1)
                B = Bq
            else:
                raise ValueError(f"Incompatible batch sizes: q {Bq}, kv {Bk}")
        else:
            B = Bq

        if mask is None:
            # Source-matrix-off is mathematically an all-zero attention bias.
            # Use SDPA to avoid materializing the dense [B, H, Nq, Nk] score tensor.
            dropout_p = self.attn_drop if self.training else 0.0
            q_fast, k_fast, v_fast = q_proj, k_proj, v_proj
            if q_fast.is_cuda and orig_dtype in (torch.float16, torch.bfloat16):
                q_fast = q_fast.to(orig_dtype)
                k_fast = k_fast.to(orig_dtype)
                v_fast = v_fast.to(orig_dtype)
            context = F.scaled_dot_product_attention(
                q_fast, k_fast, v_fast,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False,
            )
            context = context.to(dtype=q_proj.dtype)
        else:
            # Compute attention scores: Q @ K^T / sqrt(d_k)
            # q_proj: (B, num_heads, Nq, head_dim)
            # k_proj: (B, num_heads, Nk, head_dim)
            attn_scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)
            # attn_scores: (B, num_heads, Nq, Nk)

            # Apply attention bias/mask if provided
            # mask: (B, Nq, Nk) -> (B, 1, Nq, Nk)
            attn_scores = attn_scores + mask.unsqueeze(1).float()

            # FP32-safe softmax: AMP fp16 下，attention logits 范围常超过 fp16 可表示范围，
            # 且 mask=-1e4 的 row 在 fp16 softmax 中会出现 0/0 -> NaN。timm/HF 标配把
            # softmax 提到 fp32 做，再 cast 回原 dtype，对显存与吞吐影响微乎其微。
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_probs = torch.nn.functional.dropout(attn_probs, p=self.attn_drop, training=self.training)

            # Weighted sum: attn_probs @ V
            # attn_probs: (B, num_heads, Nq, Nk)
            # v_proj: (B, num_heads, Nk, head_dim)
            context = torch.matmul(attn_probs, v_proj)  # (B, num_heads, Nq, head_dim)

        # Transpose back and reshape: (B, num_heads, Nq, head_dim) -> (B, Nq, C)
        context = context.transpose(1, 2).reshape(B, Nq, self.embed_dim)

        # Output projection
        if context.is_cuda:
            with torch.amp.autocast(device_type='cuda', enabled=False):
                context = self.out_proj(context.float())
        else:
            context = self.out_proj(context.float())
        context = self.proj_drop(context)
        return context.to(orig_dtype)

class DTEMMergeOnly(DTEMBlock):
    """
    DTEM merge-only helper (no attention/MLP parameters).
    Reuses DTEMBlock's selection + merge logic without creating unused params.
    """

    def __init__(self):
        nn.Module.__init__(self)

class LocalEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 local_depth: int = 4, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 dtem_feat_dim=None, dtem_window_size: int = None, dtem_t: int = 1,
                 total_merge_local: int = 0, use_softkmax: bool = False, swa_size: int = None,
                 local_block_window: int = 16, metric_grad_scale: float = 0.1,
                 source_trace_mode: str = "center"):
        super().__init__()

        if local_depth <= 0:
            raise ValueError("local_depth must be >= 1")

        self.local_depth = local_depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.metric_grad_scale = metric_grad_scale

        # 基础 ViT 结构只用于 patch_embed / cls_token / pos_embed / norm_pre / patch_drop。
        # 注意：trailing self.vit.norm 在下方被替换为 nn.Identity()——见后续注释。
        self.vit = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                                     depth=0, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=True, num_classes=0,
                                     drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate)

        # 修订（2026-05-06）：把 LocalEncoder 末端的 trailing LN 摘掉。
        # 背景：CLSHybridToMeModel 把 LocalEncoder 4 层 + LatentEncoder 8 层串成一个 12 层
        #     pre-norm ViT 残差流。timm 的 VisionTransformer 在 blocks 之后默认会跑一次
        #     ``self.vit.norm``（trailing LN）；如果保留它，等价于"在 12 层 ViT 残差流的第 4↔5
        #     层中段强行插了一个 LayerNorm"，把前 4 层累积的残差幅度信息抹掉，仅保留方向。
        #     与 DeiT-S 12 层全局 ViT 的对照实验（c100_branch_a_deit_aligned vs
        #     c100_deit_200e）下，这条多余的 LN 是 ~1pp 级的劣化来源之一。
        # 修复：把 self.vit.norm 替换成 nn.Identity()。这样 LocalEncoder.forward 末端的
        #     ``x_out = self.vit.norm(x_merge)`` 退化为恒等映射，``x_out = x_merge`` 即原始
        #     blocks 输出（或 r>0 时的 DTEM merge 后 raw 特征），LatentEncoder block 5 接收
        #     的就是 raw 残差，与 DeiT block 5 看到的输入语义一致。最终 head 之前的
        #     ``latent.vit.norm`` 仍然保留（与 DeiT 末端 LN 对齐）。
        # 兼容性：
        #   - ``_tie_shared_embeddings`` 中 ``vit_a.norm = vit_b.norm`` 现在变成 Identity↔
        #     Identity 的 tie，行为合法。
        #   - 旧 ckpt（含 ``local.vit.norm.weight/bias``）通过 strict=False 加载时这两个 key
        #     会落到 unexpected_keys，干净忽略；不影响 blocks/metric_layers/latent 等主体
        #     权重的加载。
        #   - timm 预训练 DeiT 的 ``norm.weight/bias`` 是为"after block 12"训练的 LN，本来
        #     被错误地灌进了"after block 4"的位置（``load_pt_weights`` 不区分这一点），改成
        #     Identity 后这条潜在不一致也一并清掉。
        self.vit.norm = nn.Identity()

        # 修订（2026-05-05，应用户要求）：统一使用 LocalBlock（windowed local attention），
        # 关闭原来的 ``total_merge_local==0 -> TimmBlock`` 全局分支。
        # 目的：让分支 A（lambda_local=1, total_merge_local=0）和分支 B（lambda_local>1）的
        # local.vit.blocks 在结构与参数 shape 上完全一致，从而可以在
        # CLSDualBranchHybridToMeModel._tie_shared_embeddings 中直接 tie。
        # 副作用：分支 A 不再等价于 12 层全局 ViT/DeiT；它的语义改为
        #         "4 层 windowed local attention + r=0 不合并 token + 8 层 latent self-attn"。
        # 兼容性保留：字段 self.use_global_attn 仍然存在，但硬编码为 False，避免外部代码读到
        #             None 报错。如果未来需要恢复"全局 attn 单分支"路径，把下一行改回
        #             ``total_merge_local == 0`` 即可，配合不要 tie blocks。
        self.use_global_attn = False

        dpr = torch.linspace(0, drop_path_rate, local_depth).tolist()
        self.vit.blocks = nn.ModuleList([
            LocalBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                drop_path=dpr[i],
                local_window=local_block_window,
            )
            for i in range(local_depth)
        ])

        # DTEM metric head（每层一个）
        self.metric_dim = self._resolve_metric_dim(embed_dim, num_heads, dtem_feat_dim)
        self.metric_layers = nn.ModuleList([
            nn.Linear(embed_dim, self.metric_dim) for _ in range(local_depth)
        ])

        # 仅复用 DTEM 的 merge 逻辑（attention 不参与）
        self.merge_block = DTEMMergeOnly()

        window_size = dtem_window_size if dtem_window_size is not None else 0
        use_source_matrix = source_trace_mode in ("matrix", "detached")
        self._tome_info = {
            "r": None,
            "size": None,
            "source_matrix": None,
            "total_merge": total_merge_local,
            "trace_source": use_source_matrix,
            "prop_attn": True,
            "class_token": True,
            "distill_token": False,
            "source_tracking_mode": "matrix" if use_source_matrix else "none",
            "k2": None,
            "tau1": 1.0,
            "tau2": 30.0,
            "feat_dim": self.metric_dim,
            "window_size": window_size,
            "t": dtem_t,
            "use_softkmax": use_softkmax,
            "swa_size": swa_size,
            "local_depth": local_depth,
            "source_trace_mode": source_trace_mode,
        }
        # 共享 info 给 merge block
        self.merge_block._tome_info = self._tome_info
        # 兼容旧路径读取
        self.vit._tome_info = self._tome_info
        self.default_r = total_merge_local // max(local_depth, 1)

    @staticmethod
    def _resolve_metric_dim(embed_dim: int, num_heads: int, dtem_feat_dim):
        if dtem_feat_dim is not None:
            return dtem_feat_dim
        head_dim = embed_dim // num_heads
        return head_dim if embed_dim < 1024 else 2 * head_dim

    def _aggregate_with_source_matrix(self, x, size, source_matrix):
        if source_matrix is None:
            return x
        center = self._tome_info["source_matrix_center"]
        width = self._tome_info["source_matrix_width"]
        B, N, C = x.shape
        device = x.device

        # Vectorized: avoid Python for-loop over width (major speed bottleneck)
        base_positions = torch.arange(N, device=device)
        pos_offsets = torch.arange(width, device=device, dtype=torch.long) - center
        pos_all = base_positions.unsqueeze(1) + pos_offsets.unsqueeze(0)  # (N, width)
        pos_clamped = pos_all.clamp(0, N - 1)
        valid = (pos_all >= 0) & (pos_all < N)  # (N, width)

        weight = source_matrix * valid.to(x.dtype).unsqueeze(0)  # (B, N, width)
        index = pos_clamped.unsqueeze(0).unsqueeze(-1).expand(B, N, width, C).long()
        x_expanded = x.unsqueeze(2).expand(-1, -1, width, -1)  # (B, N, width, C)
        gathered = torch.gather(x_expanded, 1, index)  # (B, N, width, C)
        summed = (gathered * weight.unsqueeze(-1)).sum(dim=2)  # (B, N, C)

        if size is not None:
            denom = size.squeeze(-1).clamp(min=1e-6).unsqueeze(-1)
        else:
            denom = source_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return summed / denom

    @staticmethod
    def _uses_source_matrix(source_trace_mode: str) -> bool:
        return source_trace_mode in ("matrix", "detached")

    def _prepare_trace_for_forward(self):
        source_trace_mode = self._tome_info.get("source_trace_mode", "center")
        use_source_matrix = self._uses_source_matrix(source_trace_mode)

        # Keep the legacy flags aligned with the explicit mode so older merge
        # paths cannot silently re-enable dense source tracing.
        self._tome_info["trace_source"] = use_source_matrix
        self._tome_info["source_tracking_mode"] = "matrix" if use_source_matrix else "none"
        self._tome_info["source_matrix"] = None
        self._tome_info["token_center"] = None
        self._tome_info.pop("token_center_of_mass", None)
        self._tome_info.pop("source_matrix_center", None)
        self._tome_info.pop("source_matrix_width", None)
        return source_trace_mode, use_source_matrix

    def _finalize_trace_for_forward(self, source_matrix):
        source_trace_mode = self._tome_info.get("source_trace_mode", "center")
        use_source_matrix = self._uses_source_matrix(source_trace_mode)
        if not use_source_matrix and source_matrix is not None:
            raise RuntimeError(
                f"source_matrix was allocated with source_trace_mode={source_trace_mode!r}; "
                "center/none modes must keep dense source tracing disabled."
            )
        if not use_source_matrix:
            self._tome_info.pop("source_matrix_center", None)
            self._tome_info.pop("source_matrix_width", None)
        self._tome_info["source_matrix"] = source_matrix if use_source_matrix else None
        return self._tome_info["source_matrix"]

    def forward(self, x):
        x = self.vit.patch_embed(x)  # automatically inserted cls_token after patch_embed
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        n = x.shape[1]
        x_layers = []
        for local_blk in self.vit.blocks:
            x = local_blk(x)
            x_layers.append(x)
        if not x_layers:
            raise RuntimeError("LocalEncoder requires at least one local block.")
        x_embed = x_layers[-1]
        x_merge = x_embed
        r_list = parse_r(
            self.local_depth,
            self.default_r,
            self._tome_info.get("total_merge", None),
        )
        self._tome_info["r"] = r_list
        self._tome_info["size"] = torch.ones_like(x[..., 0:1])
        self._prepare_trace_for_forward()
        self._tome_info["token_counts_local"] = []

        size = self._tome_info["size"]
        source_matrix = None

        for i, layer_x in enumerate(x_layers):
            x_metric = self._aggregate_with_source_matrix(layer_x, size, source_matrix)
            s = self.metric_grad_scale
            metric_input = x_metric * s + x_metric.detach() * (1 - s)
            metric = self.metric_layers[i](metric_input)
            r = r_list[i] if i < len(r_list) else 0

            x_merge, size, n, _, source_matrix = self.merge_block._merge_train(
                x_merge, size, r, n, {"metric": metric}, source_matrix
            )

            self._tome_info["size"] = size
            self._tome_info["token_counts_local"].append(x_merge.shape[1])

        x_out = self.vit.norm(x_merge)
        self._finalize_trace_for_forward(source_matrix)
        return x_out, x_embed, self._tome_info["size"], self._tome_info


class LatentEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 depth=12, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 source_tracking_mode='map', prop_attn=True, window_size=None, use_naive_local=False, r: int = 2,
                 use_tome: bool = True):
        super().__init__()
        self.use_tome = use_tome
        self.vit = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                                     depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=True, num_classes=0, class_token=False, global_pool='',
                                     drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate)
        # 统一在 HybridToMeModel 中进行 apply_patch（去除未使用占位字段）

        # LatentEncoder receives pre-embedded tokens, so these parameters are removed
        self.vit.pos_embed.requires_grad = False
        self.vit.patch_embed = nn.Identity()
        # LatentEncoder should not use cls_token
        if hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None:
            del self.vit.cls_token
        self.vit.num_prefix_tokens = 0

    def forward(self, x, size):
        if not self.use_tome:
            token_counts_latent = []
            for blk in self.vit.blocks:
                x = blk(x)
                token_counts_latent.append(x.shape[1])
            x = self.vit.norm(x)
            info = {
                "source_map": None,
                "token_counts_latent": token_counts_latent,
                "size": size,
            }
            return x, size, info
        # 重置跨 batch 的踪迹与屏蔽，避免状态泄漏
        # self.vit._tome_info["token_mask_for_dtem"] = None
        self.vit._tome_info["r"] = parse_r(len(self.vit.blocks), self.vit._tome_info["r"], self.vit._tome_info.get("total_merge", None))
        self.vit._tome_info["size"] = size
        self.vit._tome_info["source_map"] = None
        self.vit._tome_info["source_matrix"] = None
        self.vit._tome_info["token_counts_latent"] = []
        # print(f"self.vit._tome_info: {self.vit._tome_info}")
        
        # 检查cls_token判断
        # has_cls_token = hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None
        # num_prefix_tokens = getattr(self.vit, 'num_prefix_tokens', 0)
        # print(f"[LatentEncoder] has_cls_token: {has_cls_token}, num_prefix_tokens: {num_prefix_tokens}")

        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            self.vit._tome_info["token_counts_latent"].append(x.shape[1])
            # print(f"blk._tome_info: {blk._tome_info}")
        x = self.vit.norm(x)
        return x, self.vit._tome_info["size"], self.vit._tome_info


class HybridToMeModel(nn.Module):
    
    arch_zoo = {
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 768,
                         'local_depth': 4,
                         'latent_depth': 8,
                         'num_heads': 12,
                         'mlp_ratio': 4.0
                        }),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 384,
                         'local_depth': 4,
                         'latent_depth': 8,
                         'num_heads': 6,
                         'mlp_ratio': 4.0
                        }),
        **dict.fromkeys(['s_ext', 'small_extend'],
                        {'embed_dims': 384,
                         'local_depth': 4,
                         'latent_depth': 12,
                         'num_heads': 6,
                         'mlp_ratio': 4.0
                        }),
    }  # yapf: disable

    def __init__(self, 
                 arch='base',
                 img_size=224, 
                 patch_size=16, 
                 dtem_feat_dim=None, 
                 tome_window_size=None, 
                 tome_use_naive_local=False,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 num_classes=1000, 
                 dtem_window_size: int = None, 
                 dtem_r: int = 2,
                 dtem_t: int = 1,
                 lambda_local: float = 2.0,
                 total_merge_latent: int = 4,
                 use_softkmax: bool = False,
                 local_block_window: int = 16,
                 pretrained=None,
                 pretrained_type: str = 'vit',
                 load_full_pretrained: bool = True,
                 freeze_local_encoder: bool = False,
                 swa_size: int = None,
                 metric_grad_scale: float = 0.1,
                 soft_topk: bool = False,
                 soft_topk_aux_weight: float = 0.3,
                 local_depth: int = None,
                 latent_depth: int = None,
                 source_trace_mode: str = "center",
                 **kwargs):
        super().__init__()

        # arch setups
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = dict(self.arch_zoo[arch])
            self.arch = arch.split("-")[0]
        else:
            raise ValueError("Wrong setups.")
        if local_depth is not None:
            self.arch_settings['local_depth'] = int(local_depth)
        if latent_depth is not None:
            self.arch_settings['latent_depth'] = int(latent_depth)
        if self.arch_settings['local_depth'] <= 0:
            raise ValueError("local_depth must be >= 1")
        if self.arch_settings['latent_depth'] < 0:
            raise ValueError("latent_depth must be >= 0")
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = self.arch_settings['embed_dims']
        self.num_heads = self.arch_settings['num_heads']
        self.mlp_ratio = self.arch_settings['mlp_ratio']
        self.local_depth = self.arch_settings['local_depth']
        self.latent_depth = self.arch_settings['latent_depth']

        # ------ DETM setups ------ #
        self.dtem_feat_dim = dtem_feat_dim
        self.dtem_window_size = dtem_window_size
        
        # 计算 total_merge_local: N * (lambda - 1) / lambda
        num_patches = (img_size // patch_size) ** 2
        self.total_merge_local = int(num_patches * (lambda_local - 1) / lambda_local)
        self.lambda_local = lambda_local
        
        self.total_merge_latent = total_merge_latent
        self.tome_window_size = tome_window_size
        self.dtem_t = dtem_t
        # self.dtem_r = dtem_r
        self.tome_use_naive_local = bool(tome_use_naive_local)
        self.use_softkmax = use_softkmax
        self.local_block_window = local_block_window
        self.soft_topk = soft_topk
        self.soft_topk_aux_weight = soft_topk_aux_weight
        self.metric_grad_scale = metric_grad_scale
        if source_trace_mode not in ("matrix", "detached", "center", "none"):
            raise ValueError("source_trace_mode must be one of: matrix, detached, center, none")
        self.source_trace_mode = source_trace_mode

        # ------ Linear ------ #
        self.num_classes = num_classes

        self.local = LocalEncoder(
            self.img_size,
            self.patch_size,
            self.embed_dim,
            self.num_heads,
            self.mlp_ratio,
            local_depth=self.local_depth,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            dtem_feat_dim=self.dtem_feat_dim,
            dtem_window_size=self.dtem_window_size,
            dtem_t=self.dtem_t,
            total_merge_local=self.total_merge_local,
            use_softkmax=self.use_softkmax,
            swa_size=swa_size,
            local_block_window=self.local_block_window,
            metric_grad_scale=self.metric_grad_scale,
            source_trace_mode=self.source_trace_mode,
        )
        self.latent = LatentEncoder(self.img_size, self.patch_size, self.embed_dim, self.num_heads, self.mlp_ratio,
                                    depth = self.latent_depth, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                    source_tracking_mode = 'map',
                                    prop_attn = True, 
                                    window_size = self.tome_window_size, 
                                    use_naive_local = self.tome_use_naive_local,
                                    r = self.total_merge_latent // max(self.latent_depth, 1)
                                ) if self.latent_depth > 0 else None

        self.head = nn.Linear(self.embed_dim, self.num_classes)
        
        # Cross attention 强制需要
        self.encode_cross_attention = MyCrossAttention(self.embed_dim, self.num_heads)
        self.decode_cross_attention = MyCrossAttention(self.embed_dim, self.num_heads)

        trunc_normal_(self.head.weight, std=.02)
        nn.init.zeros_(self.head.bias)

        # Zero-init cross-attention output projections so residual starts as identity
        nn.init.zeros_(self.encode_cross_attention.out_proj.weight)
        nn.init.zeros_(self.encode_cross_attention.out_proj.bias)
        nn.init.zeros_(self.decode_cross_attention.out_proj.weight)
        nn.init.zeros_(self.decode_cross_attention.out_proj.bias)

        self.swa_size = swa_size

        # 统一 apply_patch
        self._apply_patches(self.dtem_feat_dim, self.dtem_window_size, self.dtem_t, 
                            self.total_merge_local, self.tome_window_size, self.tome_use_naive_local, self.total_merge_latent, self.use_softkmax, self.swa_size)
        
        # Load pretrained weights if provided
        if pretrained:
            self._load_full_pretrained_weights(pretrained, img_size, pretrained_type, load_full_pretrained)

            # Freeze local encoder if requested (for SFT latent encoder only)
        if freeze_local_encoder:
            self.freeze_local_encoder()

    def _load_full_pretrained_weights(self, pretrained, img_size, pretrained_type='vit', load_full=True):
        """
        Load pretrained weights from timm ViT/DeiT model.
        
        Args:
            pretrained: Model name or True for auto-detection
            img_size: Image size
            pretrained_type: 'vit' or 'deit'
            load_full: If True, load full weights (Local + Latent). If False, only load Local Encoder weights.
        
        When load_full=True (default), splits the weights between Local Encoder and Latent Encoder:
        - Local Encoder (first local_depth blocks): loads blocks.0 to blocks.(local_depth-1)
        - Latent Encoder (remaining blocks): loads blocks.local_depth to blocks.(local_depth+latent_depth-1) 
          (mapped to blocks.0 to blocks.(latent_depth-1))
        
        For example, DeiT-S has 12 blocks:
        - load_full=True: Local Encoder (4 blocks) loads blocks.0-3, Latent Encoder (8 blocks) loads blocks.4-11
        - load_full=False: Only Local Encoder (4 blocks) loads blocks.0-3
        """
        import traceback
        from opentome.models.utils import load_pt_weights
        
        try:
            from timm.models import create_model
            
            # Determine model name
            if isinstance(pretrained, str):
                model_name = pretrained
            else:
                model_prefix = 'deit' if pretrained_type.lower() == 'deit' else 'vit'
                if self.embed_dim == 768 and self.num_heads == 12:
                    model_name = f'{model_prefix}_base_patch16_224'
                elif self.embed_dim == 384 and self.num_heads == 6:
                    model_name = f'{model_prefix}_small_patch16_224'
                elif self.embed_dim == 192 and self.num_heads == 3:
                    model_name = f'{model_prefix}_tiny_patch16_224'
                else:
                    print(f"[HybridToMeModel] Warning: Cannot auto-determine pretrained model for embed_dim={self.embed_dim}, num_heads={self.num_heads}. Specify model name explicitly.")
                    return
            
            load_mode_str = "full" if load_full else "local only"
            print(f"[HybridToMeModel] Loading {load_mode_str} pretrained weights from timm model: {model_name}")
            
            # Create pretrained model and extract weights
            pretrained_model = create_model(model_name, pretrained=True, img_size=img_size, num_classes=0)
            pretrained_state = pretrained_model.state_dict()
            
            # Load first local_depth blocks to Local Encoder
            print(f"[HybridToMeModel] Loading blocks [0, {self.local_depth}) to Local Encoder...")
            load_pt_weights(
                target_vit=self.local.vit,
                pretrained_state=pretrained_state,
                start_block=0,
                end_block=self.local_depth,
                verbose=True
            )
            
            # Load remaining blocks to Latent Encoder (if load_full=True and Latent Encoder exists)
            if load_full and self.latent is not None and self.latent_depth > 0:
                total_pretrained_depth = self.local_depth + self.latent_depth
                print(f"[HybridToMeModel] Loading blocks [{self.local_depth}, {total_pretrained_depth}) to Latent Encoder...")
                load_pt_weights(
                    target_vit=self.latent.vit,
                    pretrained_state=pretrained_state,
                    start_block=self.local_depth,
                    end_block=total_pretrained_depth,
                    verbose=True
                )
            elif not load_full:
                print(f"[HybridToMeModel] Skipping Latent Encoder weights (load_full=False)")
            
            print(f"[HybridToMeModel] Successfully loaded {load_mode_str} pretrained weights from {model_name}")
                    
        except Exception as e:
            print(f"[HybridToMeModel] ERROR: Failed to load pretrained weights: {e}")
            print(f"[HybridToMeModel] Exception traceback:")
            traceback.print_exc()
            # Don't raise, just warn - allow model to continue without pretrained weights

    def freeze_local_encoder(self):
        """
        Freeze all parameters in the local encoder (including local_blocks if exists).
        This is useful for SFT (Supervised Fine-Tuning) where only the latent encoder is trained.
        """
        print("[HybridToMeModel] Freezing Local Encoder parameters...")
        frozen_params = 0
        total_params = 0
        
        # Freeze local encoder (including local_blocks)
        for name, param in self.local.named_parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            total_params += param.numel()
        
        print(f"[HybridToMeModel] Frozen {frozen_params:,} parameters in Local Encoder")
        print(f"[HybridToMeModel] Total Local Encoder parameters: {total_params:,}")
        
        # Print trainable parameters summary
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_model_params = sum(p.numel() for p in self.parameters())
        print(f"[HybridToMeModel] Model trainable parameters: {trainable_params:,} / {total_model_params:,}")
    
    def unfreeze_local_encoder(self):
        """
        Unfreeze all parameters in the local encoder.
        """
        print("[HybridToMeModel] Unfreezing Local Encoder parameters...")
        unfrozen_params = 0
        
        for name, param in self.local.named_parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
        
        print(f"[HybridToMeModel] Unfrozen {unfrozen_params:,} parameters in Local Encoder")
        
        # Print trainable parameters summary
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_model_params = sum(p.numel() for p in self.parameters())
        print(f"[HybridToMeModel] Model trainable parameters: {trainable_params:,} / {total_model_params:,}")

    def _apply_patches(self, dtem_feat_dim, dtem_window_size, dtem_t, total_merge_local, tome_window_size, tome_use_naive_local, total_merge_latent, use_softkmax, swa_size):
        # LocalEncoder 内部已完成 merge 配置，仅同步关键信息
        if hasattr(self.local, "_tome_info"):
            self.local._tome_info["total_merge"] = total_merge_local
            self.local._tome_info["local_depth"] = self.local.local_depth
        
        if self.latent is not None and len(self.latent.vit.blocks) > 0:
            tome_r_per_layer = total_merge_latent//max(len(self.latent.vit.blocks),1)
            from opentome.timm.tome import tome_apply_patch
            tome_apply_patch(self.latent.vit, trace_source=True, prop_attn=True, window_size=tome_window_size,
                            use_naive_local=tome_use_naive_local, r=tome_r_per_layer
                        )
            self.latent.vit._tome_info["total_merge"] = total_merge_latent

    def set_compression_lambda(self, lambda_local: float):
        """Update the effective local compression ratio in-place (curriculum training).

        total_merge_local and the local encoder's merge budget are recomputed so the
        next forward pass merges/keeps tokens according to the new lambda. Safe to
        call between epochs; also call on EMA copies since ``_tome_info`` lives on
        the module instance and is not part of ``state_dict``.
        """
        lambda_local = float(lambda_local)
        if lambda_local < 1.0:
            raise ValueError(f"lambda_local must be >= 1.0, got {lambda_local}")
        num_patches = (self.img_size // self.patch_size) ** 2
        total_merge_local = int(num_patches * (lambda_local - 1) / lambda_local)
        self.lambda_local = lambda_local
        self.total_merge_local = total_merge_local
        if hasattr(self.local, "_tome_info"):
            self.local._tome_info["total_merge"] = total_merge_local
        self.local.default_r = total_merge_local // max(self.local.local_depth, 1)
        return total_merge_local

    def forward_ori(self,x):
        x = self.local.forward(x)
        x = self.latent.forward(x[0], None)
        cls_token_repr = x[0][:, 0]
        logits = self.head(cls_token_repr)
        aux = {}
        return logits, aux

    @staticmethod
    def _resolve_token_center_of_mass(info_local, x_local, size_local, source_matrix, device, batch_size):
        token_center = info_local.get("token_center", None)
        if token_center is not None:
            center_of_mass = token_center.to(device=device, dtype=torch.float32)
        elif source_matrix is not None:
            with torch.no_grad():
                center = info_local["source_matrix_center"]
                width = info_local["source_matrix_width"]
                B_sm, N_sm = source_matrix.shape[0], source_matrix.shape[1]
                i_positions = torch.arange(N_sm, device=device).unsqueeze(0).expand(B_sm, -1)
                offset_relative = torch.arange(width, device=device, dtype=torch.float32) - center
                weighted_offset = (source_matrix.float() * offset_relative.view(1, 1, -1)).sum(dim=-1)
                denom = size_local[..., 0].detach().float().clamp(min=1e-6)
                center_of_mass = i_positions.float() + weighted_offset / denom
        else:
            N_tokens = x_local.shape[1]
            center_of_mass = torch.arange(N_tokens, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)

        info_local["token_center_of_mass"] = center_of_mass
        return center_of_mass

    @staticmethod
    def _build_source_attention_bias(source_matrix, info_local, topk_in_full, batch_size, k, L_full, device):
        if source_matrix is None:
            return None

        with torch.no_grad():
            center = info_local["source_matrix_center"]
            width = info_local["source_matrix_width"]
            bias = torch.full((batch_size, k + 1, L_full), -1e4, device=device, dtype=torch.float32)
            bias[:, 0, :] = 0.0

            source_for_topk = torch.gather(
                source_matrix, 1,
                topk_in_full.unsqueeze(-1).expand(-1, -1, width)
            ).float()
            offset_range = torch.arange(width, device=device).view(1, 1, -1)
            j_positions = topk_in_full.unsqueeze(-1) + (offset_range - center)
            valid_mask = (j_positions >= 0) & (j_positions < L_full)
            log_source = torch.where(
                source_for_topk > 1e-10,
                torch.log(source_for_topk.clamp(min=1e-10)),
                torch.full_like(source_for_topk, -1e4)
            )
            log_source_masked = torch.where(valid_mask, log_source, torch.full_like(log_source, -1e4))
            j_positions_safe = torch.where(valid_mask, j_positions, torch.zeros_like(j_positions))
            bias[:, 1:, :].scatter_(2, j_positions_safe, log_source_masked)
            return bias


    def forward(self, x):
        B = x.shape[0]
        device = x.device
        num_patches = self.local.vit.patch_embed.num_patches
        L_full = num_patches + self.local.vit.num_prefix_tokens

        # 阶段1：LocalEncoder（DTEM软合并 + 踪迹）
        x_local, x_embed, size_local, info_local = self.local(x)
        source_matrix = info_local.get("source_matrix", None) # [B, N, width], width = 2 * window_size * local_depth + 1
        if self.source_trace_mode in ("center", "none") and source_matrix is not None:
            raise RuntimeError(
                f"source_matrix is non-empty under source_trace_mode={self.source_trace_mode!r}."
            )

        center_of_mass = self._resolve_token_center_of_mass(
            info_local, x_local, size_local, source_matrix, device, B
        )
        k = L_full - info_local["total_merge"] - 1
        token_strength = size_local[..., 0] 
        token_strength_no_cls = token_strength[:,1:]  # 去掉CLS token
        # 确保k在有效范围内
        if k <= 0 or k > token_strength_no_cls.shape[1]:
            k = token_strength_no_cls.shape[1]
        
        with torch.no_grad():
            topk_vals, topk_indices = torch.topk(token_strength_no_cls.detach(), k, dim=1, largest=True, sorted=False)
            topk_com = torch.gather(center_of_mass[:, 1:], 1, topk_indices)
            sorted_order = torch.argsort(topk_com, dim=1)
            sorted_topk_indices = torch.gather(topk_indices, 1, sorted_order)

        topk_in_full = sorted_topk_indices + 1
        topk_x_trace = torch.gather(x_local, 1, topk_in_full.unsqueeze(-1).expand(-1, -1, x_local.shape[-1]))
        topk_size_trace = torch.gather(size_local, 1, topk_in_full.unsqueeze(-1).expand(-1, -1, size_local.shape[-1]))
        topk_x = torch.cat([x_local[:, :1], topk_x_trace], dim=1)
        topk_size = torch.cat([size_local[:, :1, 0], topk_size_trace.squeeze(-1)], dim=-1).unsqueeze(-1)

        size_trace = topk_size
        bias = self._build_source_attention_bias(
            source_matrix, info_local, topk_in_full, B, k, L_full, device
        )

        x_trace = self.encode_cross_attention(topk_x, x_embed, mask=bias) + topk_x

        x_latent, size_latent, info_latent = self.latent(x_trace, size_trace)
        token_map_tome = info_latent.get("source_map", None)
        x_restore_tome = token_unmerge_from_map(x_latent, token_map_tome)
        # 阶段4：恢复（ToMe unmerge）
        # Up Sample
        x_out = self.decode_cross_attention(x_embed, x_restore_tome)
        
        cls_token_repr = x_out[:, 0]

        logits = self.head(cls_token_repr)

        aux = {"token_counts_local": info_local.get("token_counts_local", None)}
        return logits, aux


class CLSHybridToMeModel(HybridToMeModel):
    def __init__(self, *args, remove_decoder_cross_attention=False, **kwargs):
        super().__init__(*args, **kwargs)

        if remove_decoder_cross_attention:
            if hasattr(self, 'decode_cross_attention'):
                del self.decode_cross_attention

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        num_patches = self.local.vit.patch_embed.num_patches
        L_full = num_patches + self.local.vit.num_prefix_tokens

        x_local, x_embed, size_local, info_local = self.local(x)
        source_matrix = info_local.get("source_matrix", None) # [B, N, width], width = 2 * window_size * local_depth + 1
        if self.source_trace_mode in ("center", "none") and source_matrix is not None:
            raise RuntimeError(
                f"source_matrix is non-empty under source_trace_mode={self.source_trace_mode!r}."
            )

        center_of_mass = self._resolve_token_center_of_mass(
            info_local, x_local, size_local, source_matrix, device, B
        )
        k = L_full - info_local["total_merge"] - 1
        token_strength = size_local[..., 0]
        token_strength_no_cls = token_strength[:,1:]
        if k <= 0 or k > token_strength_no_cls.shape[1]:
            k = token_strength_no_cls.shape[1]

        soft_sel = None
        if self.soft_topk and self.total_merge_local > 0:
            soft_sel = ThreTopK(token_strength_no_cls, k, temperature=30.0)

        with torch.no_grad():
            topk_vals, topk_indices = torch.topk(token_strength_no_cls.detach(), k, dim=1, largest=True, sorted=False)
            topk_com = torch.gather(center_of_mass[:, 1:], 1, topk_indices)
            sorted_order = torch.argsort(topk_com, dim=1)
            sorted_topk_indices = torch.gather(topk_indices, 1, sorted_order)

        topk_in_full = sorted_topk_indices + 1
        topk_x_trace = torch.gather(x_local, 1, topk_in_full.unsqueeze(-1).expand(-1, -1, x_local.shape[-1]))
        topk_size_trace = torch.gather(size_local, 1, topk_in_full.unsqueeze(-1).expand(-1, -1, size_local.shape[-1]))

        if soft_sel is not None:
            soft_w_sel = torch.gather(soft_sel, 1, sorted_topk_indices)
            ste_w = 1.0 + (soft_w_sel - soft_w_sel.detach())
            topk_x_trace = topk_x_trace * ste_w.unsqueeze(-1)

        topk_x = torch.cat([x_local[:, :1], topk_x_trace], dim=1)
        topk_size = torch.cat([size_local[:, :1, 0], topk_size_trace.squeeze(-1)], dim=-1).unsqueeze(-1)

        size_trace = topk_size
        bias = self._build_source_attention_bias(
            source_matrix, info_local, topk_in_full, B, k, L_full, device
        )

        x_trace = self.encode_cross_attention(topk_x, x_embed, mask=bias) + topk_x

        x_latent, size_latent, info_latent = self.latent(x_trace, size_trace)

        cls_token_repr = x_latent[:, 0]
        main_logits = self.head(cls_token_repr)

        if soft_sel is not None and self.training and self.soft_topk_aux_weight > 0:
            soft_w_norm = soft_sel / soft_sel.sum(dim=1, keepdim=True).clamp(min=1e-6)
            aux_repr = (x_local[:, 1:] * soft_w_norm.unsqueeze(-1)).sum(dim=1)
            aux_logits = self.head(aux_repr)
            logits = main_logits + self.soft_topk_aux_weight * aux_logits
        else:
            logits = main_logits

        aux = {
            "token_counts_local": info_local.get("token_counts_local", None),
            # Routing/feature distillation hooks (see in1k_trainer.py):
            # - token_strength_no_cls: differentiable DTEM size mass per original
            #   patch position (soft merge keeps all tokens in spatial order), so
            #   index i aligns with teacher patch i on the same grid.
            # - topk_patch_indices: 0-based original-patch indices selected into
            #   the latent encoder (sorted by center of mass).
            "token_strength_no_cls": token_strength_no_cls,
            "topk_patch_indices": sorted_topk_indices,
            "retained_tokens": int(k),
            "cls_feature": cls_token_repr,
            "latent_tokens": x_latent[:, 1:],
        }
        return logits, aux


class CLSDualBranchHybridToMeModel(nn.Module):
    """P1 第二步：分支 A（无空间降采样）+ 分支 B（压缩 MergeNet）+ 轻量融合。

    - 共享 ``branch_b.local.vit`` 的 patch/pos/cls 与 ``branch_a.local.vit``（避免优化器重复计数，见 ``parameters``）。
    - ``forward(..., active_branch='both'|'a'|'b')``：交替训练时可只跑单分支以省显存。
    """

    def __init__(
        self,
        arch='small',
        img_size=224,
        patch_size=16,
        num_classes=1000,
        fusion_type='cat_linear',
        branch_b_lambda_local=None,
        branch_b_total_merge_latent=None,
        branch_b_dtem_window_size=None,
        branch_b_use_softkmax=None,
        branch_b_swa_size=None,
        pretrained=False,
        freeze_branch_a=False,
        **common_kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_type = fusion_type

        kw_a = dict(common_kwargs)
        kw_a['lambda_local'] = 1.0
        kw_a['total_merge_latent'] = 0

        kw_b = _with_mergenet_branch_b_defaults(common_kwargs)
        if branch_b_lambda_local is not None:
            kw_b['lambda_local'] = float(branch_b_lambda_local)
        if branch_b_total_merge_latent is not None:
            kw_b['total_merge_latent'] = int(branch_b_total_merge_latent)
        # Branch-B specific overrides only when explicitly provided.
        if branch_b_dtem_window_size is not None:
            kw_b['dtem_window_size'] = branch_b_dtem_window_size
        if branch_b_use_softkmax is not None:
            kw_b['use_softkmax'] = bool(branch_b_use_softkmax)
        if branch_b_swa_size is not None:
            kw_b['swa_size'] = branch_b_swa_size

        self.branch_a = CLSHybridToMeModel(
            arch=arch,
            remove_decoder_cross_attention=True,
            pretrained=False,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            **kw_a,
        )
        self.branch_b = CLSHybridToMeModel(
            arch=arch,
            remove_decoder_cross_attention=True,
            pretrained=False,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            **kw_b,
        )

        if pretrained:
            self.branch_a._load_full_pretrained_weights(
                pretrained, img_size,
                pretrained_type=common_kwargs.get('pretrained_type', 'vit'),
                load_full=common_kwargs.get('load_full_pretrained', True),
            )

        self._tie_shared_embeddings()

        if fusion_type == 'cat_linear':
            self.fusion_head = nn.Linear(2 * num_classes, num_classes)
            with torch.no_grad():
                W = torch.zeros(num_classes, 2 * num_classes)
                idx = torch.arange(num_classes)
                W[idx, idx] = 0.5
                W[idx, idx + num_classes] = 0.5
                self.fusion_head.weight.copy_(W)
                self.fusion_head.bias.zero_()
        elif fusion_type == 'scalar_blend':
            self.branch_mix_logit = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        if freeze_branch_a:
            self.freeze_branch_a()

    def _tie_shared_embeddings(self):
        """让 branch_a 与 branch_b 在以下五组参数上 tied（共享同一份权重 & 反传梯度合并）：

        (1) 低层视觉先验：``patch_embed`` / ``cls_token`` / ``pos_embed`` / ``norm_pre``
            两路看到的 token 化方式与位置编码必须一致，否则 fusion 输出无可比性。

        (2) ``local.vit.blocks``（4 层 LocalBlock，windowed local attention）
            前置约束：``LocalEncoder`` 已统一 ``self.use_global_attn = False``，
            两路结构与参数 shape 完全一致才能 tie。共享后两个分支的局部语义
            学习被合并到同一组权重上，仅在 token 调度（A 不合并 / B 合并 75%）
            上分化。

        (3) ``local.metric_layers``（DTEM metric heads，每层一个 Linear）
            两路对"哪些 token 重要"的判断必须一致，让 fusion 时两路 logits 是基
            于同一种"重要性观点"得到的。

        (4) ``latent``（LatentEncoder，整 8 层 LatentBlock + norm）
            两路最终都要在「同一个 latent 空间」收敛才能让 fusion_head 学到的
            线性组合具备语义可解释性。共享 latent 是双分支真正不同于"两个独立
            模型 ensemble"的核心机制。

        (5) ``encode_cross_attention`` / ``decode_cross_attention``
            分支 A 在 ``total_merge_local==0 and total_merge_latent==0`` 时直接
            ``x_trace = topk_x`` 跳过 encode_cross_attention，所以 A 路径不会前向
            也不会反传梯度；分支 B 正常使用。decode_cross_attention 在 CLS 路径
            两路都不用（CLS 走 latent->cls->head）。tie 一下纯粹去掉 dead 副本，
            行为完全等价。

        不共享（保留独立）：
          - ``head``（CLS 头）：硬约束。tie 会破坏双分支训练协议——logits_a 与
            logits_b 通过同一线性层从不同 cls 计算得到，会让 L_A / L_B 在最终
            输出层强耦合，并使 fusion 退化为"单分类器多视角集成"，失去"两个
            独立分类器 + fusion 学组合权重"的语义。
        """
        branch_a, branch_b = self.branch_a, self.branch_b
        vit_a = branch_a.local.vit
        vit_b = branch_b.local.vit

        # (1) 低层视觉先验
        vit_a.patch_embed = vit_b.patch_embed
        if hasattr(vit_b, 'cls_token') and vit_b.cls_token is not None:
            vit_a.cls_token = vit_b.cls_token
        vit_a.pos_embed = vit_b.pos_embed
        if hasattr(vit_b, 'norm_pre') and vit_b.norm_pre is not None:
            vit_a.norm_pre = vit_b.norm_pre

        # (2) LocalEncoder 4 层 LocalBlock
        # 前置：LocalEncoder.__init__ 已强制 use_global_attn=False，两路结构一致
        assert not branch_a.local.use_global_attn and not branch_b.local.use_global_attn, \
            "[_tie_shared_embeddings] 期望两路 LocalEncoder 都使用 LocalBlock（use_global_attn=False）"
        vit_a.blocks = vit_b.blocks
        # 同步 LocalEncoder 顶层 norm。
        # 修订（2026-05-06）：``LocalEncoder.__init__`` 已把 ``self.vit.norm`` 替换为
        # ``nn.Identity()``，此处 tie 实际上是 Identity↔Identity 的同步，仅为对称性保留，
        # 不影响 forward 行为；保留这段代码是为了在未来若把 LocalEncoder 末端 LN 改回
        # 真 LN 时仍能正确 tie（避免分支 A/B 用了不同 LN 权重）。
        if hasattr(vit_b, 'norm') and vit_b.norm is not None:
            vit_a.norm = vit_b.norm

        # (3) DTEM metric heads
        branch_a.local.metric_layers = branch_b.local.metric_layers

        # (4) LatentEncoder（仅当两路都有 latent 时才 tie；latent_depth>0 默认成立）
        if branch_a.latent is not None and branch_b.latent is not None:
            branch_a.latent = branch_b.latent

        # (5) cross attention 模块（A 路径不前向，B 路径才用；共享纯省 dead 参数）
        if hasattr(branch_a, 'encode_cross_attention') and hasattr(branch_b, 'encode_cross_attention'):
            branch_a.encode_cross_attention = branch_b.encode_cross_attention
        if hasattr(branch_a, 'decode_cross_attention') and hasattr(branch_b, 'decode_cross_attention'):
            branch_a.decode_cross_attention = branch_b.decode_cross_attention

    def freeze_branch_a(self):
        for p in self.branch_a.parameters():
            p.requires_grad = False

    def unfreeze_branch_a(self):
        for p in self.branch_a.parameters():
            p.requires_grad = True

    def load_branch_a_from_single_model_checkpoint(self, path, map_location='cpu',
                                                    align_branch_b_head=True,
                                                    fusion_init='prefer_a'):
        """加载「单分支 A」checkpoint（``hybridtomevit_small_cls_branch_a`` 训练得到）到 ``branch_a``。

        warm-start 排异反应「残余项」修复（2026-05-15）：
        ----------------------------------------------------------------------
        前置上下文（与 ``in1k_trainer`` 中 ``_split_optimizer_for_branch_a`` /
        ``_clip_params_for_step`` 配合）：lr_scale=0 + clip-local 已经把"共享 encoder
        被 L_b 噪声反传更新"和"unfreeze 时 AdamW v_t shock"这两条根因封死。但
        eval_top1 在 stage2 epoch 0 仍然会从 stage1 的 ~73 跌到 ~60，原因在于
        ``CLSDualBranchHybridToMeModel.__init__`` 默认状态下的两个"残余项"：

        (α) ``branch_b.head`` 与 ``branch_a.head`` **不** tie（参见
            ``_tie_shared_embeddings`` 的"不共享"注释——硬约束，否则会破坏双分支
            训练协议）。因此 warm-start 把 stage1 的 head 灌进 ``branch_a.head``
            后，``branch_b.head`` 仍是 ``trunc_normal_(std=0.02)`` 随机初始化。
            forward 时 ``logits_b = head_b(cls_b_repr)`` 是高熵噪声，
            ``L_b ≈ ln(num_classes) ≈ 4.6`` 主导总 loss；即便 lr_scale=0 冻住
            shared encoder 的"更新"，head_b 的 SGD 步会在前 ~几个 batch 把
            ``|W_b|`` 推大，``logits_b`` 开始 confidently 预测错类。
        (β) ``fusion_head`` 默认 ``W[i,i]=W[i,i+nc]=0.5, bias=0``，对"两路同等
            可信"的 from-scratch 场景是合理初始化，但 warm-start 时
            ``logits_a`` 是 stage1 金标准、``logits_b`` 是噪声，50/50 融合等于
            把 a 的精度直接腰斩 ⇒ ``eval_top1`` 在 epoch 0 就从 73 掉到 60+。

        本方法在 warm-start 时一次性把两条都修了：

        Args:
            align_branch_b_head: 把刚刚加载到 ``branch_a.head`` 的 stage1 head
                权重**直接复制**到 ``branch_b.head``。两路 head 不再共享（保持
                双分支训练协议），但起点完全一致，L_b 初始值与 L_a 同量级，
                ``head_b`` 的梯度从一开始就是"如何在压缩后的 cls 上微调"而非
                "如何从随机噪声中恢复分类信号"。
            fusion_init:
                - ``'prefer_a'``（默认，warm-start 推荐）：把 fusion_head 重置为
                  ``W[i,i]=1.0`` (branch_a 部分)、``W[i,i+nc]=0.0`` (branch_b
                  部分)、``bias=0``。等价于 epoch 0 的 ``logits_fused = logits_a``，
                  eval_top1 锁死在 stage1 水平。随后 ``W[i,i+nc]`` 通过 L_fused
                  的梯度自适应吸收 branch_b 的信号。
                - ``'balanced'``：保留 __init__ 中的 0.5/0.5（适合 from-scratch 双
                  路对称训练；warm-start 不推荐，会触发上面 (β) 的问题）。
                - ``'keep'`` / ``None``：不动 fusion_head，沿用当前状态（用于
                  resume 等更复杂场景）。
        """
        raw = _torch_load_checkpoint(path, map_location=map_location)
        if isinstance(raw, dict) and 'state_dict' in raw:
            sd = raw['state_dict']
        elif isinstance(raw, dict) and 'model' in raw:
            sd = raw['model']
        elif isinstance(raw, dict) and 'state_dict' not in raw and 'model' not in raw:
            sd = raw
        else:
            sd = raw
        new_sd = {}
        for k, v in sd.items():
            k = k.replace('module.', '')
            new_sd['branch_a.' + k] = v
        missing, unexpected = self.load_state_dict(new_sd, strict=False)

        # (α) 把 stage1 head 复制到 branch_b.head，消除 logits_b 的随机噪声起点。
        # 注意：用 .data.copy_ 而非赋值同一引用——两路 head 仍是独立 nn.Linear，
        # 后续训练各自更新；只是 epoch 0 起点完全一致，避免随机 head 噪声反传
        # 污染 fusion_head / L_b 梯度。
        if align_branch_b_head and hasattr(self, 'branch_a') and hasattr(self, 'branch_b'):
            head_a = getattr(self.branch_a, 'head', None)
            head_b = getattr(self.branch_b, 'head', None)
            if (
                head_a is not None and head_b is not None
                and hasattr(head_a, 'weight') and hasattr(head_b, 'weight')
                and head_a.weight.shape == head_b.weight.shape
            ):
                with torch.no_grad():
                    head_b.weight.data.copy_(head_a.weight.data)
                    if (getattr(head_a, 'bias', None) is not None
                            and getattr(head_b, 'bias', None) is not None
                            and head_a.bias.shape == head_b.bias.shape):
                        head_b.bias.data.copy_(head_a.bias.data)

        # (β) 重置 fusion_head，让 epoch 0 的 logits_fused 等于 logits_a，
        # eval 锁在 stage1 水平。仅对 cat_linear 融合生效（scalar_blend 只有
        # 一个标量 mix logit，由 sigmoid(0)=0.5 兜底，本来就近似 50/50；
        # 如果未来要 warm-start scalar_blend，可在此把 branch_mix_logit 推到
        # 大正值，但当前训练协议默认走 cat_linear，先不动）。
        if (fusion_init == 'prefer_a' and getattr(self, 'fusion_type', None) == 'cat_linear'
                and hasattr(self, 'fusion_head')):
            with torch.no_grad():
                nc = self.num_classes
                W = torch.zeros(nc, 2 * nc, device=self.fusion_head.weight.device,
                                dtype=self.fusion_head.weight.dtype)
                idx = torch.arange(nc)
                W[idx, idx] = 1.0  # branch_a 全权
                # branch_b 部分留 0，让 L_fused 的梯度从 0 开始自适应吸收 logits_b
                self.fusion_head.weight.data.copy_(W)
                if getattr(self.fusion_head, 'bias', None) is not None:
                    self.fusion_head.bias.data.zero_()
        elif fusion_init == 'balanced' and getattr(self, 'fusion_type', None) == 'cat_linear' \
                and hasattr(self, 'fusion_head'):
            with torch.no_grad():
                nc = self.num_classes
                W = torch.zeros(nc, 2 * nc, device=self.fusion_head.weight.device,
                                dtype=self.fusion_head.weight.dtype)
                idx = torch.arange(nc)
                W[idx, idx] = 0.5
                W[idx, idx + nc] = 0.5
                self.fusion_head.weight.data.copy_(W)
                if getattr(self.fusion_head, 'bias', None) is not None:
                    self.fusion_head.bias.data.zero_()
        # 其他情况（'keep' / None / 未知值）：保留当前 fusion_head 状态不动。

        return missing, unexpected

    def parameters(self, recurse=True):
        seen = set()
        for p in nn.Module.parameters(self, recurse):
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            yield p

    def forward(self, x, active_branch='both'):
        """
        Args:
            active_branch: ``both`` | ``a`` | ``b`` — 仅前向单分支时用于 T2 省显存 / 分段阶段 1。
        """
        logits_a = logits_b = None
        aux_a = aux_b = {}

        if active_branch in ('both', 'a'):
            logits_a, aux_a = self.branch_a(x)
        if active_branch in ('both', 'b'):
            logits_b, aux_b = self.branch_b(x)

        if active_branch == 'a':
            aux = {
                'dual_branch': True,
                'logits_a': logits_a,
                'logits_b': None,
                'logits_fused': logits_a,
                'active_branch': 'a',
            }
            return logits_a, aux

        if active_branch == 'b':
            aux = {
                'dual_branch': True,
                'logits_a': None,
                'logits_b': logits_b,
                'logits_fused': logits_b,
                'active_branch': 'b',
            }
            return logits_b, aux

        if self.fusion_type == 'cat_linear':
            logits_fused = self.fusion_head(torch.cat([logits_a, logits_b], dim=-1))
        else:
            w = torch.sigmoid(self.branch_mix_logit)
            logits_fused = w * logits_a + (1.0 - w) * logits_b

        aux = {
            'dual_branch': True,
            'logits_a': logits_a,
            'logits_b': logits_b,
            'logits_fused': logits_fused,
            'active_branch': 'both',
            'token_counts_local': aux_a.get('token_counts_local'),
        }
        return logits_fused, aux


@register_model
def hybridtomevit_base(**kwargs):
    """HybridToMe ViT Base model"""
    model = HybridToMeModel(arch='base', **kwargs)
    return model

@register_model
def hybridtomevit_small(**kwargs):
    """HybridToMe ViT Small model"""
    model = HybridToMeModel(arch='small', **kwargs)
    return model

# ------ For Image Classification ------ #
@register_model
def hybridtomevit_base_cls(**kwargs):
    """HybridToMe ViT Base model"""
    model = CLSHybridToMeModel(arch='base', remove_decoder_cross_attention=True, **kwargs)
    return model

@register_model
def mergenet_small_cls(**kwargs):
    """Canonical single-branch MergeNet-B classifier."""
    kwargs = _with_mergenet_small_defaults(kwargs)
    model = CLSHybridToMeModel(arch='small', remove_decoder_cross_attention=True, **kwargs)
    return model


@register_model
def hybridtomevit_small_cls(**kwargs):
    """Compatibility alias for ``mergenet_small_cls``."""
    return mergenet_small_cls(**kwargs)


@register_model
def hybridtomevit_small_cls_branch_a(pretrained=False, num_classes=1000, **kwargs):
    """P1 分支 A：Local 4L（windowed local attention）+ Latent 8L，无空间 token 降采样。

    固定 ``lambda_local=1`` → ``total_merge_local=0``，``total_merge_latent=0``，
    与 ``CLSHybridToMeModel`` 中跳过 cross-attn 残差、全程不降采样的前向路径一致，
    用于与 DeiT-Small 同数据/增强/epoch/优化器协议对齐的对照实验。

    历史行为变更（2026-05-05）：原先 ``LocalEncoder`` 在 ``total_merge_local==0`` 时切到
    全局 ``TimmBlock``，本工厂因此曾等价于 12 层全局 ViT。为了让双分支 T1/T2/T3 能
    tie ``local.vit.blocks``，``LocalEncoder`` 已统一为 ``LocalBlock``（windowed local
    attention，``local_window=local_block_window``，默认 16）。本工厂创建出的模型
    的 attention 也随之变成 windowed local。参数命名（``norm1``/``attn.qkv``/``attn.proj``/
    ``norm2``/``mlp.fc1``/``mlp.fc2``）与原 TimmBlock 完全一致，旧 P1 第一步 ckpt 仍然
    可以 ``strict=False`` 加载，作为权重初始化使用，但 attention 行为已是 windowed。
    """
    kwargs = dict(kwargs)
    kwargs['lambda_local'] = 1.0
    kwargs['total_merge_latent'] = 0
    model = CLSHybridToMeModel(
        arch='small',
        remove_decoder_cross_attention=True,
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )
    return model


@register_model
def mergenet_small_cls_dual_ab(pretrained=False, num_classes=1000, **kwargs):
    """Dual container: branch A anchor + canonical ``mergenet_small_cls`` branch B + fusion."""
    kwargs = dict(kwargs)
    fusion_type = kwargs.pop('fusion_type', 'cat_linear')
    branch_b_lambda_local = kwargs.pop('branch_b_lambda_local', None)
    branch_b_total_merge_latent = kwargs.pop('branch_b_total_merge_latent', None)
    branch_b_dtem_window_size = kwargs.pop('branch_b_dtem_window_size', None)
    branch_b_use_softkmax = kwargs.pop('branch_b_use_softkmax', None)
    branch_b_swa_size = kwargs.pop('branch_b_swa_size', None)
    freeze_branch_a = bool(kwargs.pop('freeze_branch_a', False))
    return CLSDualBranchHybridToMeModel(
        arch='small',
        fusion_type=fusion_type,
        branch_b_lambda_local=branch_b_lambda_local,
        branch_b_total_merge_latent=branch_b_total_merge_latent,
        branch_b_dtem_window_size=branch_b_dtem_window_size,
        branch_b_use_softkmax=branch_b_use_softkmax,
        branch_b_swa_size=branch_b_swa_size,
        pretrained=pretrained,
        num_classes=num_classes,
        freeze_branch_a=freeze_branch_a,
        **kwargs,
    )


@register_model
def hybridtomevit_small_cls_dual_ab(pretrained=False, num_classes=1000, **kwargs):
    """Compatibility alias for ``mergenet_small_cls_dual_ab``."""
    return mergenet_small_cls_dual_ab(pretrained=pretrained, num_classes=num_classes, **kwargs)


@register_model
def hybridtomevit_small_cls_ext(**kwargs):
    """HybridToMe ViT Small model"""
    model = CLSHybridToMeModel(arch='s_ext', remove_decoder_cross_attention=True, **kwargs)
    return model


# python /yuchang/yk/benchmark_scaleup.py --devices cuda:0 --lengths 64000,128000,256000,512000,1024000,2048000,4096000 --num_workers 8 --model_name resnet50 --model_path /yuchang/yk/resnet50_mixup.pth


if __name__ == '__main__':
    """ Debug script for mergenet_small_cls model during development """
    from timm.models import create_model
    # Create model instance (using default parameters similar to trainer)
    print("=" * 60)
    print("Creating mergenet_small_cls model...")
    
    model = create_model(
        'mergenet_small_cls',  # model_name must be the first positional argument
        pretrained=False,
        num_classes=1000,
        img_size=224,
        patch_size=8,
        dtem_window_size=7,
        # dtem_r=2,
        dtem_t=1,
        lambda_local=4.0,
        total_merge_latent=0,
        use_softkmax=False,
        local_depth=4,
        local_block_window=32,
        tome_window_size=32,
        tome_use_naive_local=False,
        swa_size=None,
        freeze_local_encoder=False,  # Keep local encoder trainable
    )
    # Set model to eval mode for inference
    model.eval()
    print(model)
    
    # Create dummy input: 224x224x3 image -> BxCxHxW format
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Move model and input to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    
    # Forward pass
    print("\n" + "=" * 60)
    print("Running forward pass...")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    # Handle output (model returns (logits, aux) tuple)
    if isinstance(output, tuple):
        logits, aux = output
        print(f"\nOutput logits shape: {logits.shape}")
        print(f"Output logits (first 10 values): {logits[0, :10]}")
        print(f"\nAuxiliary info: {aux}")
    else:
        print(f"\nOutput shape: {output.shape}")
        print(f"Output (first 10 values): {output[0, :10]}")
