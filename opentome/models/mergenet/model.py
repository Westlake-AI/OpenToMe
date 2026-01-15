# /opentome/models/model.py

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.layers import trunc_normal_
from timm.models.registry import register_model

# from opentome.timm.tome import tome_apply_patch
from opentome.timm.dtem import dtem_apply_patch, trace_token_merge, token_unmerge_from_map_for_dtem
from opentome.tome.tome import token_unmerge_from_map, parse_r
from opentome.timm.bias_local_attn import LocalBlock

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash_attn not available, falling back to standard attention")

class MyCrossAttention(nn.Module):
    """
    Implements multi-head cross attention using Flash Attention where query and key/value sequences
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
        Bq, Nq, C = q.shape
        Bk, Nk, Ck = kv.shape
        assert C == self.embed_dim and Ck == self.embed_dim

        # Compute projections
        q_proj = self.q_proj(q)  # (Bq, Nq, C)
        k_proj = self.k_proj(kv) # (Bk, Nk, C)
        v_proj = self.v_proj(kv) # (Bk, Nk, C)

        # Reshape for multi-head: (B, N, C) -> (B, N, num_heads, head_dim)
        q_proj = q_proj.reshape(Bq, Nq, self.num_heads, self.head_dim)
        k_proj = k_proj.reshape(Bk, Nk, self.num_heads, self.head_dim)
        v_proj = v_proj.reshape(Bk, Nk, self.num_heads, self.head_dim)

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

        if FLASH_ATTN_AVAILABLE and q_proj.dtype in [torch.float16, torch.bfloat16]:
            # Use Flash Attention 2
            # flash_attn_func expects: (batch, seqlen, nheads, headdim)
            # Input is already in correct shape: (B, N, num_heads, head_dim)
            
            try:
                # Try to use flash_attn with bias support
                # In newer versions, flash_attn_func supports attn_bias parameter
                if mask is not None:
                    # Prepare attention bias
                    # Flash Attention expects bias in shape (batch, nheads, seqlen_q, seqlen_k)
                    attn_bias = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).contiguous()
                else:
                    attn_bias = None
                
                context = flash_attn_func(
                    q_proj, k_proj, v_proj,
                    dropout_p=self.attn_drop if self.training else 0.0,
                    softmax_scale=1.0 / (self.head_dim ** 0.5),
                    causal=False,
                    deterministic=not self.training,  # Ensure deterministic behavior during eval
                )  # (B, Nq, num_heads, head_dim)
                
                # Note: If your flash_attn version supports attn_bias, add it as a parameter above
                # flash_attn_func(..., attn_bias=attn_bias)
                # For versions without native bias support, we fall back when mask is present
                if mask is not None:
                    # Fallback to standard attention for mask support
                    context = self._standard_attention(q_proj, k_proj, v_proj, mask)
                    
            except Exception as e:
                # Fall back to standard attention on any error
                print(f"Flash Attention failed: {e}, falling back to standard attention")
                context = self._standard_attention(q_proj, k_proj, v_proj, mask)
        else:
            # Fall back to standard attention (wrong dtype or flash_attn not available)
            context = self._standard_attention(q_proj, k_proj, v_proj, mask)

        # Reshape back to (B, Nq, C)
        context = context.reshape(B, Nq, self.embed_dim)

        context = self.out_proj(context)
        context = self.proj_drop(context)
        return context
    
    def _standard_attention(self, q, k, v, mask=None):
        """
        Standard attention implementation as fallback
        Args:
            q, k, v: (B, N, num_heads, head_dim)
            mask: (B, Nq, Nk) or None
        Returns:
            context: (B, Nq, num_heads, head_dim)
        """
        # Transpose to (B, num_heads, N, head_dim) for matmul
        q = q.transpose(1, 2)  # (B, num_heads, Nq, head_dim)
        k = k.transpose(1, 2)  # (B, num_heads, Nk, head_dim)
        v = v.transpose(1, 2)  # (B, num_heads, Nk, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            # mask: (B, Nq, Nk) -> (B, 1, Nq, Nk)
            attn_scores = attn_scores + mask.unsqueeze(1)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = torch.nn.functional.dropout(attn_probs, p=self.attn_drop, training=self.training)
        
        # Weighted sum
        context = torch.matmul(attn_probs, v)  # (B, num_heads, Nq, head_dim)
        context = context.transpose(1, 2)  # (B, Nq, num_heads, head_dim)
        
        return context

class LocalEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 depth=4, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 feat_dim=None, window_size: int = None, r: int = 2, t: int = 1, num_classes=10,
                 num_local_blocks: int = 0, local_block_window: int = 16):
        super().__init__()
        
        # 添加额外的 LocalBlocks（在 DTEM blocks 之前）
        self.num_local_blocks = num_local_blocks
        if num_local_blocks > 0:
            self.local_blocks = nn.ModuleList([
                LocalBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    local_window=local_block_window,
                )
                for _ in range(num_local_blocks)
            ])
        else:
            self.local_blocks = None
        
        self.vit = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                                     depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=True, num_classes=0,
                                     drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate)

    def forward(self, x):
        x = self.vit.patch_embed(x)  # automatically inserted cls_token after patch_embed
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        
        # 重置跨 batch 的踪迹与屏蔽，避免状态泄漏
        # self.vit._tome_info["token_mask_for_dtem"] = None
        # 与 timm 的 ViT 对齐，启用 norm_pre
        x = self.vit.norm_pre(x)
        n = x.shape[1]
        self.vit._tome_info["r"] = parse_r(len(self.vit.blocks), self.vit.r, self.vit._tome_info.get("total_merge", None))
        self.vit._tome_info["size"] = torch.ones_like(x[..., 0:1])
        self.vit._tome_info["token_counts_local"] = []
        
        # 检查cls_token判断
        # has_cls_token = hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None
        # num_prefix_tokens = getattr(self.vit, 'num_prefix_tokens', 0)
        # print(f"[LocalEncoder] has_cls_token: {has_cls_token}, num_prefix_tokens: {num_prefix_tokens}")
        
        # 先运行额外的 LocalBlocks（不改变 size, n, source_matrix）
        if self.local_blocks is not None:
            for local_blk in self.local_blocks:
                x = local_blk(x)
        x_embed = x.clone()  # 保存未merge的embedding用于后续cross attention
        source_matrix = None  # Initialize, will be created in first block
        for i, blk in enumerate(self.vit.blocks):
            x, size, n, _, source_matrix = blk(x, self.vit._tome_info["size"], n=n, source_matrix=source_matrix)
            self.vit._tome_info["size"] = size
            self.vit._tome_info["token_counts_local"].append(x.shape[1])
        x = self.vit.norm(x)
        # Add source_matrix to info_local for return
        self.vit._tome_info["source_matrix"] = source_matrix
        return x, x_embed, self.vit._tome_info["size"], self.vit._tome_info


class LatentEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 depth=12, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 source_tracking_mode='map', prop_attn=True, window_size=None, use_naive_local=False, r: int = 2):
        super().__init__()
        self.vit = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                                     depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=True, num_classes=0, class_token=False, global_pool='',
                                     drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate)
        # 统一在 HybridToMeModel 中进行 apply_patch（去除未使用占位字段）

        # LatentEncoder receives pre-embedded tokens, so these parameters are removed
        self.vit.patch_embed = nn.Identity()
        # LatentEncoder should not use cls_token
        if hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None:
            del self.vit.cls_token
        self.vit.num_prefix_tokens = 0

    def forward(self, x, size):
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
                #  dtem_r: int = 2, 
                 dtem_t: int = 1,
                 lambda_local: float = 2.0,
                 total_merge_latent: int = 4,
                 use_softkmax: bool = False,
                 num_local_blocks: int = 0,
                 local_block_window: int = 16,
                 pretrained=None,
                 swa_size: int = None,
                 **kwargs):
        super().__init__()

        # arch setups
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
            self.arch = arch.split("-")[0]
        else:
            raise ValueError("Wrong setups.")
        
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
        self.num_local_blocks = num_local_blocks
        self.local_block_window = local_block_window

        # ------ Linear ------ #
        self.num_classes = num_classes

        self.local = LocalEncoder(self.img_size, self.patch_size, self.embed_dim, self.num_heads, self.mlp_ratio, 
                                  num_classes = self.num_classes, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                  depth = self.local_depth, 
                                  feat_dim = self.dtem_feat_dim, 
                                  window_size = self.dtem_window_size,
                                  r = self.total_merge_local // max(self.local_depth, 1), 
                                  t = self.dtem_t,
                                  num_local_blocks = self.num_local_blocks,
                                  local_block_window = self.local_block_window
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

        self.swa_size = swa_size

        # 统一 apply_patch
        self._apply_patches(self.dtem_feat_dim, self.dtem_window_size, self.dtem_t, 
                            self.total_merge_local, self.tome_window_size, self.tome_use_naive_local, self.total_merge_latent, self.use_softkmax, self.swa_size)


    def _apply_patches(self, dtem_feat_dim, dtem_window_size, dtem_t, total_merge_local, tome_window_size, tome_use_naive_local, total_merge_latent, use_softkmax, swa_size):
        # DTEM patch
        dtem_r_per_layer = total_merge_local//max(len(self.local.vit.blocks),1)
        # Cross attention 强制启用
        dtem_apply_patch(self.local.vit, feat_dim=dtem_feat_dim, trace_source=True, prop_attn=True,
                         default_r=dtem_r_per_layer, window_size=dtem_window_size, t=dtem_t, use_softkmax=use_softkmax, swa_size=swa_size)
        
        # 记录总merge数，供 parse_r 使用
        self.local.vit._tome_info["total_merge"] = total_merge_local
        self.local.vit._tome_info["local_depth"] = len(self.local.vit.blocks)
        
        if self.latent is not None and len(self.latent.vit.blocks) > 0:
            tome_r_per_layer = total_merge_latent//max(len(self.latent.vit.blocks),1)
            from opentome.timm.tome import tome_apply_patch
            tome_apply_patch(self.latent.vit, trace_source=True, prop_attn=True, window_size=tome_window_size,
                            use_naive_local=tome_use_naive_local, r=tome_r_per_layer
                        )
            self.latent.vit._tome_info["total_merge"] = total_merge_latent
    

    def forward_ori(self,x):
        x = self.local.forward(x)
        x = self.latent.forward(x[0], None)
        cls_token_repr = x[0][:, 0]
        logits = self.head(cls_token_repr)
        aux = {}
        return logits, aux


    def forward(self, x):
        B = x.shape[0]
        device = x.device
        num_patches = self.local.vit.patch_embed.num_patches
        L_full = num_patches + self.local.vit.num_prefix_tokens

        # 阶段1：LocalEncoder（DTEM软合并 + 踪迹）
        x_local, x_embed, size_local, info_local = self.local(x)
        source_matrix = info_local.get("source_matrix", None) # [B, N, width], width = 2 * window_size * local_depth + 1
        
        # Compute center of mass for each token based on source_matrix
        # 🔧 FIX: 使用 torch.no_grad() 避免将 source_matrix 计算拉入梯度图
        if source_matrix is not None:
            with torch.no_grad():
                center = info_local["source_matrix_center"]
                width = info_local["source_matrix_width"]
                B_sm, N_sm = source_matrix.shape[0], source_matrix.shape[1]
                i_positions = torch.arange(N_sm, device=device).unsqueeze(0).expand(B_sm, -1)  # (B, N)
                offset_relative = torch.arange(width, device=device, dtype=torch.float32) - center  # (width,)
                
                # Weighted relative offset: source_matrix * (offset - center)
                weighted_offset = (source_matrix * offset_relative.view(1, 1, -1)).sum(dim=-1)  # (B, N)
                
                # Center of mass = current position + weighted offset / size
                # 🔧 使用 detach() 避免 size_local 的梯度传播到这里
                token_center_of_mass = i_positions.float() + weighted_offset / size_local[..., 0].detach().clamp(min=1e-6)
            
            # Store in info_local
            info_local["token_center_of_mass"] = token_center_of_mass  # (B, N)
        
        center_of_mass = info_local["token_center_of_mass"] # [B, N]
        k = L_full - info_local["total_merge"] - 1
        token_strength = size_local[..., 0] 
        token_strength_no_cls = token_strength[:,1:]  # 去掉CLS token
        # 确保k在有效范围内
        if k <= 0 or k > token_strength_no_cls.shape[1]:
            k = token_strength_no_cls.shape[1]
        
        # 🔧 FIX: topk 和 argsort 操作不需要梯度，用 detach() 避免保留索引计算的中间结果
        with torch.no_grad():
            topk_vals, topk_indices = torch.topk(token_strength_no_cls.detach(), k, dim=1, largest=True, sorted=False)  # (B, k)
            topk_com = torch.gather(center_of_mass, 1, topk_indices)  # (B, k)
            sorted_order = torch.argsort(topk_com, dim=1)  # (B, k)
            sorted_topk_indices = torch.gather(topk_indices, 1, sorted_order)  # (B, k)
        
        # 使用索引 gather 实际的 token 和 size（这些需要梯度）
        topk_x_trace = torch.gather(x_local, 1, sorted_topk_indices.unsqueeze(-1).expand(-1, -1, x_local.shape[-1]))
        topk_size_trace = torch.gather(size_local, 1, sorted_topk_indices.unsqueeze(-1).expand(-1, -1, size_local.shape[-1]))
        topk_x = torch.cat([x_local[:, :1], topk_x_trace], dim=1)
        topk_size = torch.cat([size_local[:, :1, 0], topk_size_trace.squeeze(-1)], dim=-1).unsqueeze(-1)

        size_trace = topk_size
        
        # 构建 attention bias: log(source_matrix) 作为先验
        # 形状: [B, k+1, L_full]
        # 🔧 FIX: bias 构建过程用 torch.no_grad() 并 detach()，避免保留巨大计算图
        with torch.no_grad():
            center = info_local["source_matrix_center"]
            width = info_local["source_matrix_width"]
            
            # 初始化 bias 为大负数（表示不能 attend）
            bias = torch.full((B, k+1, L_full), -1e10, device=device, dtype=x_local.dtype)
            
            # cls token（第 0 行）不设 bias，可以 attend 所有位置
            bias[:, 0, :] = 0.0
            
            # 获取 topk tokens 在 x_local 中的实际索引（+1 因为 cls token）
            actual_indices = sorted_topk_indices + 1  # [B, k]
            
            # 从 source_matrix 中提取对应行
            # source_matrix: [B, N, width] -> source_for_topk: [B, k, width]
            source_for_topk = torch.gather(
                source_matrix, 
                1, 
                actual_indices.unsqueeze(-1).expand(-1, -1, width)
            )  # [B, k, width]
            
            # 计算每个 offset 对应的原序列位置
            # j = actual_indices[b, i] + (offset - center)
            offset_range = torch.arange(width, device=device).view(1, 1, -1)  # [1, 1, width]
            j_positions = actual_indices.unsqueeze(-1) + (offset_range - center)  # [B, k, width]
            
            # 合法性检查
            valid_mask = (j_positions >= 0) & (j_positions < L_full)  # [B, k, width]
            
            # 对 source 值取 log，零值或极小值保持为 -1e10
            log_source = torch.where(
                source_for_topk > 1e-10,
                torch.log(source_for_topk.clamp(min=1e-10)),
                torch.full_like(source_for_topk, -1e10)
            )  # [B, k, width]
            
            # 矢量化 scatter：将 log_source 填充到 bias 的对应位置
            # 对于无效位置，保持 -1e10（不改变 bias）
            log_source_masked = torch.where(valid_mask, log_source, torch.full_like(log_source, -1e10))
            
            # 将无效的 j_positions clamp 到 0（防止索引错误）
            j_positions_safe = torch.where(valid_mask, j_positions, torch.zeros_like(j_positions))
            
            # 使用 scatter_ 在最后一个维度上更新 bias[:, 1:, :]
            bias[:, 1:, :].scatter_(2, j_positions_safe, log_source_masked)
        
        # Down Sample
        x_trace = self.encode_cross_attention(topk_x, x_embed, mask=bias)
        # 阶段3：LatentEncoder（ToMe硬合并）
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

        if source_matrix is not None:
            with torch.no_grad():
                center = info_local["source_matrix_center"]
                width = info_local["source_matrix_width"]
                B_sm, N_sm = source_matrix.shape[0], source_matrix.shape[1]
                i_positions = torch.arange(N_sm, device=device).unsqueeze(0).expand(B_sm, -1)  # (B, N)
                offset_relative = torch.arange(width, device=device, dtype=torch.float32) - center  # (width,)
                weighted_offset = (source_matrix * offset_relative.view(1, 1, -1)).sum(dim=-1)  # (B, N)
                token_center_of_mass = i_positions.float() + weighted_offset / size_local[..., 0].detach().clamp(min=1e-6)
            
            # Store in info_local
            info_local["token_center_of_mass"] = token_center_of_mass  # (B, N)
        
        center_of_mass = info_local["token_center_of_mass"] # [B, N]
        k = L_full - info_local["total_merge"] - 1
        token_strength = size_local[..., 0] 
        token_strength_no_cls = token_strength[:,1:]  
        if k <= 0 or k > token_strength_no_cls.shape[1]:
            k = token_strength_no_cls.shape[1]
        
        with torch.no_grad():
            topk_vals, topk_indices = torch.topk(token_strength_no_cls.detach(), k, dim=1, largest=True, sorted=False)  # (B, k)
            topk_com = torch.gather(center_of_mass, 1, topk_indices)  # (B, k)
            sorted_order = torch.argsort(topk_com, dim=1)  # (B, k)
            sorted_topk_indices = torch.gather(topk_indices, 1, sorted_order)  # (B, k)
        
        topk_x_trace = torch.gather(x_local, 1, sorted_topk_indices.unsqueeze(-1).expand(-1, -1, x_local.shape[-1]))
        topk_size_trace = torch.gather(size_local, 1, sorted_topk_indices.unsqueeze(-1).expand(-1, -1, size_local.shape[-1]))
        topk_x = torch.cat([x_local[:, :1], topk_x_trace], dim=1)
        topk_size = torch.cat([size_local[:, :1, 0], topk_size_trace.squeeze(-1)], dim=-1).unsqueeze(-1)

        size_trace = topk_size
        with torch.no_grad():
            center = info_local["source_matrix_center"]
            width = info_local["source_matrix_width"]
            bias = torch.full((B, k+1, L_full), -1e10, device=device, dtype=x_local.dtype)
            
            bias[:, 0, :] = 0.0

            actual_indices = sorted_topk_indices + 1  # [B, k]

            source_for_topk = torch.gather(
                source_matrix, 
                1, 
                actual_indices.unsqueeze(-1).expand(-1, -1, width)
            )  # [B, k, width]
            
            offset_range = torch.arange(width, device=device).view(1, 1, -1)  # [1, 1, width]
            j_positions = actual_indices.unsqueeze(-1) + (offset_range - center)  # [B, k, width]
            
            valid_mask = (j_positions >= 0) & (j_positions < L_full)  # [B, k, width]
            log_source = torch.where(
                source_for_topk > 1e-10,
                torch.log(source_for_topk.clamp(min=1e-10)),
                torch.full_like(source_for_topk, -1e10)
            )  # [B, k, width]
            log_source_masked = torch.where(valid_mask, log_source, torch.full_like(log_source, -1e10))
            j_positions_safe = torch.where(valid_mask, j_positions, torch.zeros_like(j_positions))
            bias[:, 1:, :].scatter_(2, j_positions_safe, log_source_masked)
        
        # Down Sample
        x_trace = self.encode_cross_attention(topk_x, x_embed, mask=bias)
        x_latent, size_latent, info_latent = self.latent(x_trace, size_trace)

        # cls_token_repr = x_latent[:, 0]
        cls_token_repr = x_latent.mean(dim=1)
        logits = self.head(cls_token_repr)

        aux = {"token_counts_local": info_local.get("token_counts_local", None)}
        return logits, aux




@register_model
def hybridtomevit_base(pretrained=False, **kwargs):
    """HybridToMe ViT Base model"""
    model = HybridToMeModel(arch='base', **kwargs)
    return model

@register_model
def hybridtomevit_small(pretrained=False, **kwargs):
    """HybridToMe ViT Small model"""
    model = HybridToMeModel(arch='small', **kwargs)
    return model

# ------ For Image Classification ------ #
@register_model
def hybridtomevit_base_cls(pretrained=False, **kwargs):
    """HybridToMe ViT Base model"""
    model = CLSHybridToMeModel(arch='base', remove_decoder_cross_attention=True, **kwargs)
    return model

@register_model
def hybridtomevit_small_cls(pretrained=False, **kwargs):
    """HybridToMe ViT Small model"""
    model = CLSHybridToMeModel(arch='small', remove_decoder_cross_attention=True, **kwargs)
    return model

# python /yuchang/yk/benchmark_scaleup.py --devices cuda:0 --lengths 64000,128000,256000,512000,1024000,2048000,4096000 --num_workers 8 --model_name resnet50 --model_path /yuchang/yk/resnet50_mixup.pth


if __name__ == '__main__':
    """ Debug script for hybridtomevit_small_cls model during development """
    from timm.models import create_model
    # Create model instance (using default parameters similar to trainer)
    print("=" * 60)
    print("Creating hybridtomevit_small_cls model...")
    
    model = create_model(
        'hybridtomevit_small_cls',  # model_name must be the first positional argument
        pretrained=False,
        num_classes=1000,
        img_size=224,
        patch_size=8,
        dtem_window_size=7,
        dtem_r=2,
        dtem_t=1,
        lambda_local=4.0,
        total_merge_latent=4,
        use_softkmax=False,
        num_local_blocks=0,
        local_block_window=32,
        tome_window_size=32,
        tome_use_naive_local=False,
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
