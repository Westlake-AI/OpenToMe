# /opentome/models/model.py

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.layers import trunc_normal_

from opentome.timm.tome import tome_apply_patch
from opentome.timm.dtem import dtem_apply_patch, trace_token_merge, token_unmerge_from_map_for_dtem
from opentome.tome.tome import token_unmerge_from_map, parse_r
from opentome.timm.bias_local_attn import LocalBlock

class LocalEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 depth=4, feat_dim=None, window_size: int = None, r: int = 2, t: int = 1, num_classes=10, use_cross_attention: bool = True,
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
                                     drop_rate=0.0,attn_drop_rate=0.0,drop_path_rate=0.0,)
    def forward(self, x):
        x = self.vit.patch_embed(x)
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
        x_embed = x.clone()
        source_matrix = None  # Initialize, will be created in first block
        for i, blk in enumerate(self.vit.blocks):
            x, size, n, _, source_matrix = blk(x, self.vit._tome_info["size"], n=n, source_matrix=source_matrix)
            self.vit._tome_info["size"] = size
            self.vit._tome_info["token_counts_local"].append(x.shape[1])
        x = self.vit.norm(x)
        # Add source_matrix to info_local for return
        self.vit._tome_info["source_matrix"] = source_matrix
        return x, x_embed, self.vit._tome_info["size"], self.vit._tome_info

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
        
        # q from seq_q, k/v from seq_kv
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, mask=None):
        """
        Args:
            q: (Bq, Nq, C)      -- queries
            kv: (Bk, Nk, C)     -- keys/values
            mask: (Bq, Nq, Nk), optional -- attention mask
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

        # Reshape for multi-head: (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q_proj = q_proj.reshape(Bq, Nq, self.num_heads, self.head_dim).transpose(1,2)  # (Bq, num_heads, Nq, head_dim)
        k_proj = k_proj.reshape(Bk, Nk, self.num_heads, self.head_dim).transpose(1,2)  # (Bk, num_heads, Nk, head_dim)
        v_proj = v_proj.reshape(Bk, Nk, self.num_heads, self.head_dim).transpose(1,2)  # (Bk, num_heads, Nk, head_dim)

        # Handle broadcasting in batch dimension
        # If Bq == Bk == 1, normal
        # If Bq != Bk, repeat accordingly
        if Bq != Bk:
            if Bq == 1:
                # broadcast q over Bk
                q_proj = q_proj.expand(Bk, -1, -1, -1) # (Bk, num_heads, Nq, head_dim)
                B = Bk
            elif Bk == 1:
                # broadcast k/v over Bq
                k_proj = k_proj.expand(Bq, -1, -1, -1) # (Bq, num_heads, Nk, head_dim)
                v_proj = v_proj.expand(Bq, -1, -1, -1)
                B = Bq
            else:
                raise ValueError(f"Incompatible batch sizes: q {Bq}, kv {Bk}")
        else:
            B = Bq

        # Transpose to (B, num_heads, Nq, head_dim), (B, num_heads, Nk, head_dim)
        # Compute attention scores: Q @ K^T / sqrt(head_dim)
        attn_scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, Nq, Nk)

        if mask is not None:
            # attn_scores = attn_scores.masked_fill(mask.unsqueeze(1)==0, float("-inf"))
            attn_scores = attn_scores + mask.unsqueeze(1)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # Weighted sum over values
        context = torch.matmul(attn_probs, v_proj)  # (B, num_heads, Nq, head_dim)
        context = context.transpose(1,2).reshape(B, Nq, self.embed_dim) # (B, Nq, C)

        context = self.out_proj(context)
        context = self.proj_drop(context)
        return context

class LatentEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 depth=12, source_tracking_mode='map', prop_attn=True, window_size=None, use_naive_local=False, r: int = 2):
        super().__init__()
        self.vit = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                                     depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=True, num_classes=0,
                                     drop_rate=0.0,attn_drop_rate=0.0,drop_path_rate=0.0,)
        # 统一在 HybridToMeModel 中进行 apply_patch（去除未使用占位字段）

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
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12,
                 mlp_ratio=4.0, dtem_feat_dim=None, local_depth=4, latent_depth=12,
                 tome_window_size=None, tome_use_naive_local=False, num_classes=10, dtem_window_size: int = None, dtem_r: int = 2, dtem_t: int = 1,
                 total_merge_local: int = 8, total_merge_latent: int = 4, use_softkmax: bool = False, use_cross_attention: bool = False,
                 num_local_blocks: int = 0, local_block_window: int = 16):
        super().__init__()
        # import pdb;pdb.set_trace()
        self.local = LocalEncoder(img_size, patch_size, embed_dim, num_heads, mlp_ratio, num_classes=num_classes,
                                  depth=local_depth, feat_dim=dtem_feat_dim, window_size=dtem_window_size,
                                  r=total_merge_local//max(local_depth,1), t=dtem_t, use_cross_attention=use_cross_attention,
                                  num_local_blocks=num_local_blocks, local_block_window=local_block_window)
        self.latent = LatentEncoder(img_size, patch_size, embed_dim, num_heads, mlp_ratio,
                                    depth=latent_depth, source_tracking_mode='map',
                                    prop_attn=True, window_size=tome_window_size, use_naive_local=tome_use_naive_local,
                                    r=total_merge_latent//max(latent_depth,1)) if latent_depth>0 else None
        self.head = nn.Linear(embed_dim, num_classes)
        self.encode_cross_attention = MyCrossAttention(embed_dim, num_heads) if use_cross_attention else None
        self.decode_cross_attention = MyCrossAttention(embed_dim, num_heads) if use_cross_attention else None
        trunc_normal_(self.head.weight, std=.02)
        nn.init.zeros_(self.head.bias)
        # 统一 apply_patch
        self._apply_patches(dtem_feat_dim, dtem_window_size, dtem_t, total_merge_local, tome_window_size, tome_use_naive_local, total_merge_latent, use_softkmax, use_cross_attention)

    def _unmerge_with_source_matrix(self, x, source_matrix, center, width):
        """
        Unmerge tokens using source_matrix (vectorized).
        source_matrix[i, offset] = how much token i contributes to position i+(offset-center)
        For each position j, gather contributions from all i where i+(offset-center)=j
        
        Args:
            x: (B, N, C) - token representations
            source_matrix: (B, N, width) - contribution matrix
            center: int - center offset in source_matrix
            width: int - width of source_matrix
        
        Returns:
            x_unmerged: (B, N, C) - unmerged representations
        """
        B, N, C = x.shape
        device = x.device
        
        # Vectorized approach: for each (i, offset), scatter to j = i + offset - center
        # x: (B, N, C) -> (B, N, 1, C)
        # source_matrix: (B, N, width) -> (B, N, width, 1)
        x_expanded = x.unsqueeze(2)  # (B, N, 1, C)
        source_expanded = source_matrix.unsqueeze(-1)  # (B, N, width, 1)
        
        # Weighted contributions: (B, N, width, C)
        weighted_x = x_expanded * source_expanded  # (B, N, width, C)
        
        # Compute target positions: j = i + offset - center
        i_idx = torch.arange(N, device=device).view(1, -1, 1)  # (1, N, 1)
        offset_idx = torch.arange(width, device=device).view(1, 1, -1)  # (1, 1, width)
        j_idx = i_idx + offset_idx - center  # (1, N, width)
        
        # Valid mask
        valid = (j_idx >= 0) & (j_idx < N)  # (1, N, width)
        j_idx_clamped = j_idx.clamp(0, N - 1)  # (1, N, width)
        
        # Apply valid mask
        weighted_x_masked = weighted_x * valid.unsqueeze(0).unsqueeze(-1).to(x.dtype)  # (B, N, width, C)
        source_masked = source_matrix * valid.to(x.dtype)  # (B, N, width)
        
        # Scatter to target positions using 3D indexing
        x_unmerged = torch.zeros(B, N, C, device=device, dtype=x.dtype)
        weight_sum = torch.zeros(B, N, device=device, dtype=x.dtype)
        
        # Expand indices for scatter_add
        j_idx_expanded = j_idx_clamped.expand(B, -1, -1).unsqueeze(-1).expand(-1, -1, -1, C)  # (B, N, width, C)
        
        # Flatten B and N*width dimensions for scatter_add
        # x_unmerged: (B, N, C) -> view as (B, N*C) to use 1D scatter
        # weighted_x_masked: (B, N, width, C) -> reshape to (B, N*width*C)
        
        # Use linear indexing: batch_idx * (N*C) + j_idx * C + c_idx
        batch_idx = torch.arange(B, device=device).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        c_idx = torch.arange(C, device=device).view(1, 1, 1, -1)  # (1, 1, 1, C)
        
        linear_idx = batch_idx * (N * C) + j_idx_expanded * C + c_idx  # (B, N, width, C)
        
        # Flatten and scatter
        x_flat = x_unmerged.reshape(-1)  # (B*N*C,)
        x_flat.scatter_add_(0, linear_idx.reshape(-1), weighted_x_masked.reshape(-1).to(x.dtype))
        x_unmerged = x_flat.reshape(B, N, C)
        
        # Scatter weights
        j_idx_for_weights = j_idx_clamped.expand(B, -1, -1)  # (B, N, width)
        weight_sum.scatter_add_(1, j_idx_for_weights.reshape(B, -1), source_masked.reshape(B, -1).to(x.dtype))
        
        # Normalize
        weight_sum = weight_sum.unsqueeze(-1).clamp(min=1e-6)
        x_unmerged = x_unmerged / weight_sum
        
        return x_unmerged
    
    def _apply_patches(self, dtem_feat_dim, dtem_window_size, dtem_t, total_merge_local, tome_window_size, tome_use_naive_local, total_merge_latent, use_softkmax, use_cross_attention):
        # DTEM patch
        dtem_r_per_layer = total_merge_local//max(len(self.local.vit.blocks),1)
        dtem_apply_patch(self.local.vit, feat_dim=dtem_feat_dim, trace_source=True, prop_attn=True,
                         default_r=dtem_r_per_layer, window_size=dtem_window_size, t=dtem_t, use_softkmax=use_softkmax, use_cross_attention=use_cross_attention)
        # 记录总merge数，供 parse_r 使用
        self.local.vit._tome_info["total_merge"] = total_merge_local
        self.local.vit._tome_info["local_depth"] = len(self.local.vit.blocks)
        if self.latent is not None and len(self.latent.vit.blocks) > 0:
            tome_r_per_layer = total_merge_latent//max(len(self.latent.vit.blocks),1)
            from opentome.timm.tome import tome_apply_patch
            tome_apply_patch(self.latent.vit, trace_source=True, prop_attn=True, window_size=tome_window_size,
                                use_naive_local=tome_use_naive_local, r=tome_r_per_layer)
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
        # import pdb;pdb.set_trace()
        x_local, x_embed,size_local, info_local = self.local(x)
        source_matrix = info_local.get("source_matrix", None) # [B, N, width], width = 2 * window_size * local_depth + 1
        
        # Compute center of mass for each token based on source_matrix
        if source_matrix is not None:
            center = info_local["source_matrix_center"]
            width = info_local["source_matrix_width"]
            B_sm, N_sm = source_matrix.shape[0], source_matrix.shape[1]
            i_positions = torch.arange(N_sm, device=device).unsqueeze(0).expand(B_sm, -1)  # (B, N)
            offset_relative = torch.arange(width, device=device, dtype=torch.float32) - center  # (width,)
            
            # Weighted relative offset: source_matrix * (offset - center)
            weighted_offset = (source_matrix * offset_relative.view(1, 1, -1)).sum(dim=-1)  # (B, N)
            
            # Center of mass = current position + weighted offset / size
            token_center_of_mass = i_positions.float() + weighted_offset / size_local[..., 0].clamp(min=1e-6)
            
            # Store in info_local
            info_local["token_center_of_mass"] = token_center_of_mass  # (B, N)
        
        # import pdb;pdb.set_trace()


        center_of_mass = info_local["token_center_of_mass"] # [B, N]
        if self.encode_cross_attention is not None:
            k = L_full - info_local["total_merge"] - 1
            if k <= 0:
                k = size_local.shape[1]
            token_strength = size_local[..., 0] 
            token_strength = token_strength[:,1:]
            topk_vals, topk_indices = torch.topk(token_strength, k, dim=1, largest=True, sorted=False)  # (B, k)
            topk_com = torch.gather(center_of_mass, 1, topk_indices)  # (B, k)
            sorted_order = torch.argsort(topk_com, dim=1)  # (B, k)
            sorted_topk_indices = torch.gather(topk_indices, 1, sorted_order)  # (B, k)
            topk_x_trace = torch.gather(x_local, 1, sorted_topk_indices.unsqueeze(-1).expand(-1, -1, x_local.shape[-1]))
            topk_size_trace = torch.gather(size_local, 1, sorted_topk_indices.unsqueeze(-1).expand(-1, -1, size_local.shape[-1]))
            topk_x = torch.cat([x_local[:, :1], topk_x_trace], dim=1)
            topk_size = torch.cat([size_local[:, :1, 0], topk_size_trace.squeeze(-1)], dim=-1).unsqueeze(-1)

            size_trace = topk_size
            
            # 构建 attention bias: log(source_matrix) 作为先验
            # 形状: [B, k+1, L_full]
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
            # import pdb;pdb.set_trace()
            x_trace = self.encode_cross_attention(topk_x, x_embed, mask=bias)
        else:
            raise ValueError("Cross attention is not supported")
        #     if self.training:
        #         x_trace, size_trace, token_map_for_dtem_compact = trace_token_merge(
        #             x_local, size_local, token_map_for_dtem
        #         )
        #     else:
        #         x_trace, size_trace = x_local, size_local
        #         token_map_for_dtem_compact = token_map_for_dtem
        # 阶段3：LatentEncoder（ToMe硬合并）
        x_latent, size_latent, info_latent = self.latent(x_trace, size_trace)
        token_map_tome = info_latent.get("source_map", None)
        x_restore_tome = token_unmerge_from_map(x_latent, token_map_tome)
        # import pdb;pdb.set_trace()
        # 阶段4：两阶段恢复（ToMe -> DTEM）
        # x_out_full_seq = token_unmerge_from_map_for_dtem(
        #     x_restore_tome, token_map_for_dtem_compact, T_full=L_full
        # )
        # import pdb;pdb.set_trace()
        if self.decode_cross_attention is not None:
            # Up Sample
            # x_out = torch.rand_like(x_local)
            x_restore_tome = self.decode_cross_attention(x_embed, x_restore_tome)
        else:
            raise ValueError("Cross attention is not supported")
            # x_restore_tome = token_unmerge_from_map_for_dtem(
            #     x_restore_tome, token_map_for_dtem_compact, T_full=L_full
            # )
        if self.training and source_matrix is not None:
            x_out_full_seq = self._unmerge_with_source_matrix(
                x_restore_tome, source_matrix, 
                info_local["source_matrix_center"],
                info_local["source_matrix_width"]
            )
        else:
            x_out_full_seq = x_restore_tome
        
        cls_token_repr = x_out_full_seq[:, 0]
        logits = self.head(cls_token_repr)

        aux = {"token_counts_local": info_local.get("token_counts_local", None)}
        return logits, aux
# python /yuchang/yk/benchmark_scaleup.py --devices cuda:0 --lengths 64000,128000,256000,512000,1024000,2048000,4096000 --num_workers 8 --model_name resnet50 --model_path /yuchang/yk/resnet50_mixup.pth