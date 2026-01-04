# ------ jinxin modified ------ #
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath, use_fused_attn
from timm.models.vision_transformer import VisionTransformer, LayerScale
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock
from opentome.timm.thetopk import ThreTopK
from functools import partial

from opentome.tome.tome import (
    bipartite_soft_matching,
    merge_source_matrix,
    merge_source_map,
    merge_wavg,
    parse_r,
)
from opentome.timm import Attention, Block
from opentome.timm.bias_local_attn import biased_local_attention


# ---------------------- dtem 专用通用 map/trace 工具 ---------------------- #
# def merge_source_map_for_dtem(current_level_map, source_map):
#     if source_map is None:
#         return current_level_map
#     return torch.gather(current_level_map, dim=1, index=source_map)

@torch.no_grad()
def trace_token_merge(x, size, token_map):
    """
    将 token_map 所指定的分组进行加权聚合，返回合并后的 x/size 以及压缩后的映射：
    - x: [B, L, C]
    - size: [B, L] 或 [B, L, 1] 或 None（None 表示等权重）
    - token_map: [B, L]，每个原 token 合并到的目标 group 索引
    返回 (x_merged [B, M_max, C], size_merged [B, M_max, 1], compact_token_map [B, L])
    """
    B, L, C = x.shape
    device = x.device
    dtype = x.dtype

    if size is None:
        size_local = torch.ones(B, L, device=device, dtype=dtype)
    else:
        size_local = size.squeeze(-1) if size.ndim == 3 else size

    offset = torch.arange(B, device=device, dtype=torch.long).view(B, 1) * L
    global_token_map = token_map + offset

    unique_global_indices, inverse_indices = torch.unique(global_token_map.flatten(), return_inverse=True)
    batch_indices = torch.div(unique_global_indices, L, rounding_mode='floor')

    perm = torch.argsort(unique_global_indices)
    sorted_batch_indices = batch_indices[perm]
    is_new_batch = torch.cat([torch.tensor([True], device=device), sorted_batch_indices[1:] != sorted_batch_indices[:-1]])
    batch_group_ids = torch.cumsum(is_new_batch.long(), dim=0) - 1
    segment_starts = torch.where(is_new_batch)[0]
    local_rank_sorted = torch.arange(len(unique_global_indices), device=device) - segment_starts[batch_group_ids]
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(unique_global_indices), device=device)
    local_rank_unsorted = local_rank_sorted[inv_perm]
    compact_token_map = local_rank_unsorted[inverse_indices].view(B, L)

    counts_per_batch = torch.bincount(batch_indices, minlength=B)
    M_max = counts_per_batch.max().item()

    index = compact_token_map.unsqueeze(-1).expand_as(x)
    x_w = x * size_local.unsqueeze(-1)

    num = torch.zeros(B, M_max, C, device=device, dtype=dtype)
    den = torch.zeros(B, M_max, device=device, dtype=dtype)

    num.scatter_add_(dim=1, index=index, src=x_w)
    den.scatter_add_(dim=1, index=compact_token_map, src=size_local)

    den_clamped = torch.clamp(den, min=torch.finfo(den.dtype).eps)
    x_merged = num / den_clamped.unsqueeze(-1)
    size_merged = den.unsqueeze(-1)

    return x_merged, size_merged, compact_token_map

def token_unmerge_from_map_for_dtem(merged_tokens, token_map_compact, T_full):
    """
    将长度 L-k 的 merged_tokens 还原到原长度 L：
    - merged_tokens: [B, L-k, C]
    - token_map_compact: [B, L]，每个原 token 在合并后序列中的位置
    - T_full: 原长度 L
    返回: [B, L, C]
    """
    B, Lk, C = merged_tokens.shape
    device = merged_tokens.device
    b_idx = torch.arange(B, device=device)[:, None].expand(B, T_full)
    return merged_tokens[b_idx, token_map_compact]

class DTEMLinear(nn.Linear):
    def __init__(self, qkv_layer, feat_dim):
        super().__init__(in_features=qkv_layer.weight.shape[1], out_features=qkv_layer.weight.shape[0] + feat_dim, bias=True)
        # qkv
        self.qkv_layer = qkv_layer

        # metric
        self.feat_dim = feat_dim
        self.metric_layer = nn.Linear(qkv_layer.weight.shape[-1], feat_dim)

        # copy
        self.update()

    @torch.no_grad()
    def update(self):
        # Ensure weights are on the same device
        device = self.qkv_layer.weight.device
        
        # Move metric_layer to the same device as qkv_layer
        if self.metric_layer.weight.device != device:
            self.metric_layer = self.metric_layer.to(device)
        
        if self.weight.device != device:
            self.weight.data = self.weight.data.to(device)
            self.bias.data = self.bias.data.to(device)
        
        # qkv -> self
        self.weight.data[:-self.feat_dim].copy_(self.qkv_layer.weight.data)
        self.bias.data[:-self.feat_dim].copy_(self.qkv_layer.bias.data)
        
        # metric_layer -> self
        self.weight.data[-self.feat_dim:].copy_(self.metric_layer.weight.data)
        self.bias.data[-self.feat_dim:].copy_(self.metric_layer.bias.data)

    def train(self, mode=True):
        if mode is False:   # if eval
            self.update()
        return super().train(mode)

    def forward(self, input: torch.Tensor):
        if not self.training:
            # Ensure weights are on the same device as input
            if self.weight.device != input.device:
                self.update()
            out = F.linear(input, self.weight, self.bias)
            return out[..., :-self.feat_dim], out[..., -self.feat_dim:]
        
        # training
        # Ensure metric_layer is on the same device as input
        if self.metric_layer.weight.device != input.device:
            self.metric_layer = self.metric_layer.to(input.device)
        
        out1 = self.qkv_layer(input)  # Shape: (B, N, 3 * num_heads * head_dim)
        out2 = self.metric_layer(input.detach())  # Shape: (B, N, feat_dim)
        return out1, out2


try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except Exception:
    FLASH_ATTN_AVAILABLE = False


"""
    timm - deit patch
"""
class DTEMAttention(Attention):
    def patch(self, feat_dim):
        if feat_dim is not None:
            out_dim = feat_dim
        else:
            dim = self.head_dim * self.num_heads
            out_dim = self.head_dim if dim < 1024 else 2 * self.head_dim 
        # add metric_layer
        self.qkv = DTEMLinear(self.qkv, out_dim)
    
    def forward(self, x, size=None):    # x:(B, N, C), size:(B, N) or (B, N, 1)
        B, N, C = x.shape
        out1, out2 = self.qkv(x)
        qkv = out1.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)     # B, H, N, head_dim
        q, k = self.q_norm(q), self.k_norm(k)
        # import pdb;pdb.set_trace()

        # fp32 for softmax computation
        q, k, v = q.type(torch.float32), k.type(torch.float32), v.type(torch.float32)
        with torch.cuda.amp.autocast(dtype=torch.float32, enabled=True):
            q = q * self.scale
            window_size = self._tome_info.get("window_size")
            if window_size is not None and window_size > 0:
                # 尝试使用 triton 实现；失败回退到 naive/flash
                x_bnc = None
                # import pdb;pdb.set_trace()
                try:
                    from .local_attn_triton import local_attn_sizebias_bhnd
                    x_bnhd = local_attn_sizebias_bhnd(q, k, v, size, window_size, softmax_scale=1.0)  # (B, N, H, D)
                    x_bnc = x_bnhd.reshape(B, N, self.num_heads * self.head_dim)
                    x_bnc = self.attn_drop(x_bnc)
                except Exception:
                    # 先试 flash，再退回 naive
                    # x_bnc = self._flash_local_attention(q, k, v, size, window_size, x_dtype=x.dtype) if FLASH_ATTN_AVAILABLE else None
                    x_bnc = None
                    if FLASH_ATTN_AVAILABLE and size is not None:
                        try:
                            # 处理 bias：size 可能是 (B, N) 或 (B, N, 1)
                            size_log = size.squeeze(-1).log() if size.ndim == 3 else size.log()
                            # 调用 flash attention
                            x_bhnd = biased_local_attention(
                                q, k, v, 
                                bias=size_log,
                                local_window=window_size,
                                dropout_p=self.attn_drop.p,
                                training=self.training,
                                x_dtype=x.dtype
                            )  # 返回 (B, H, N, D)
                            # 转换为 (B, N, H*D)
                            x_bnc = x_bhnd.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
                        except Exception as e:
                            # Flash attention 失败，x_bnc 保持 None
                            pass
                    
                    if x_bnc is None:
                        x_bnc = self._naive_local_attention(q, k, v, size, window_size, x_dtype=x.dtype)
            else:
                # 全局注意力路径
                attn = q @ k.transpose(-2, -1)
                if size is None or (not self._tome_info["r"]): # for MAE
                    attn = attn.softmax(dim=-1)
                else:   # as in DynamicViT
                    _attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
                    size_ = size if size is not None else torch.ones(B, N, 1, device=x.device, dtype=torch.float32)
                    _attn = _attn.exp_() * size_[:, None, None, :, 0].type(torch.float32)
                    attn = _attn / _attn.sum(dim=-1, keepdim=True)
                attn = self.attn_drop(attn)
                # (B, H, N, D) -> (B, N, C)
                x_bnc = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, N, self.num_heads * self.head_dim)

        x = x_bnc
        x = self.proj(x)
        x = self.proj_drop(x)
        metric = dict(
            q = q,
            k = k,
            v = v,
            x = x,
            metric = out2
        )
        return x, metric

    def _naive_local_attention(self, q, k, v, size, window_size, x_dtype=torch.float32):
        """
        无依赖的局部窗口注意力回退实现。
        q,k,v: (B, H, N, D)
        size: (B, N, 1) or (B, N) or None
        返回: (B, N, C)
        """
        B, H, N, D = q.shape
        BH = B * H

        q_flat = q.transpose(1, 2).reshape(BH, N, D)
        k_flat = k.transpose(1, 2).reshape(BH, N, D)
        v_flat = v.transpose(1, 2).reshape(BH, N, D)

        padded_k = F.pad(k_flat, (0, 0, window_size, window_size))
        padded_v = F.pad(v_flat, (0, 0, window_size, window_size))

        k_windows = padded_k.unfold(dimension=1, size=2 * window_size + 1, step=1)  # (BH, N, win, D)
        v_windows = padded_v.unfold(dimension=1, size=2 * window_size + 1, step=1)  # (BH, N, win, D)
        k_windows = k_windows.transpose(-1, -2)
        v_windows = v_windows.transpose(-1, -2)
        q_reshaped = q_flat.unsqueeze(2)  # (BH, N, 1, D)
        attn = (q_reshaped @ k_windows.transpose(-1, -2)).squeeze(2)  # (BH, N, win)

        if size is not None:
            size_log = size.log().squeeze(-1) if size.ndim == 3 else size.log()
            size_bias_bh = size_log.unsqueeze(1).expand(-1, H, -1).reshape(BH, N)
            padded_size_bias = F.pad(size_bias_bh, (window_size, window_size), mode='constant', value=-1e9)
            size_bias_windows = padded_size_bias.unfold(dimension=1, size=2 * window_size + 1, step=1)
            attn = attn + size_bias_windows

        win_indices = torch.arange(-window_size, window_size + 1, device=q.device).view(1, -1)
        q_indices = torch.arange(N, device=q.device).view(-1, 1)
        abs_k_pos = q_indices + win_indices  # (N, win)
        mask = (abs_k_pos >= 0) & (abs_k_pos < N)
        attn = attn.masked_fill(~mask.unsqueeze(0), float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_out_flat = (attn.unsqueeze(2) @ v_windows).squeeze(2)  # (BH, N, D)
        attn_out = attn_out_flat.view(B, H, N, D).transpose(1, 2)  # (B, N, H, D)
        x_processed = attn_out.reshape(B, N, H * D).to(x_dtype)
        return x_processed




class DTEMBlock(Block):
    
    def _select(self, k, a, b, a_orig_idx=None, b_orig_idx=None):
        # import pdb;pdb.set_trace()
        """
        统一的选择逻辑：支持全局或局部窗口匹配，并支持跨层屏蔽。
        a: (B, Na, C), b: (B, Nb, C)
        a_orig_idx: (B, Na) - a tokens 的原始位置索引
        b_orig_idx: (B, Nb) - b tokens 的原始位置索引
        返回 assign: 
            - 全局模式 (window_size <= 0): (B, Na, Nb)
            - 局部窗口模式 (window_size > 0): (B, Na, 2*window_size+1) 狭长矩阵
        """
        EPSILON = torch.finfo(torch.float32).eps
        B, Na, C = a.shape
        _, Nb, _ = b.shape
        device = a.device

        window_size = self._tome_info.get("window_size")
        if window_size is None or window_size <= 0:
            scores = a @ b.transpose(-1, -2)
            _idx = scores.argsort(dim=-1, descending=True)
            _x = scores.gather(dim=-1, index=_idx)
            _x = _x / (self._tome_info.get("tau1", 1.0))
            if self._tome_info.get("use_softkmax")==False:
                # print("use_softkmax is False")
                # print(f"_x: {_x.shape}")
                # print(f"k: {k}")
                # print(f"self._tome_info['tau2']: {self._tome_info['tau2']}")
                khot = torch.zeros_like(_x)
                _x_iter = _x.clone()  # Clone to avoid inplace modification
                for _ in range(k):
                    onehot_approx = F.softmax(_x_iter.view(B, -1) / (self._tome_info.get("tau2", 0.1)), dim=-1).view(B, Na, Nb)
                    khot += onehot_approx
                    khot_mask = torch.clamp(1 - onehot_approx.sum(dim=-1, keepdim=True), min=EPSILON)
                    _x_iter = _x_iter + torch.log(khot_mask)
            else:
                # print("use_softkmax is True")
                # print(f"_x: {_x.shape}")
                # print(f"k: {k}")
                # print(f"self._tome_info['tau2']: {self._tome_info['tau2']}")
                _x_energy = torch.logsumexp(_x, dim=-1).exp()
                row_khot = ThreTopK(_x_energy, k, self._tome_info["tau2"])
                khot = F.softmax(_x, dim=-1) * row_khot.unsqueeze(-1)

            tmp = torch.clamp(khot.sum(dim=-1, keepdim=True).detach() - 1, min=0.) + 1.
            nkhot = khot / tmp
            nkhot = nkhot.to(scores.dtype)
            assign = torch.zeros_like(scores).scatter_reduce(-1, _idx, nkhot, reduce='sum')
        else:
            padded_b = F.pad(b, (0, 0, window_size, window_size))
            b_windows_unfolded = padded_b.unfold(dimension=1, size=2 * window_size + 1, step=1)
            b_windows = b_windows_unfolded.permute(0, 1, 3, 2)
            a_reshaped = a.unsqueeze(2)
            local_scores = (a_reshaped * b_windows).sum(dim=-1) / (self._tome_info.get("tau1", 1.0))
            
            # Mask out invalid positions (outside [0, Nb) range)
            a_indices = torch.arange(Na, device=device).view(1, -1, 1)
            window_offsets = torch.arange(-window_size, window_size + 1, device=device).view(1, 1, -1)
            b_indices = a_indices + window_offsets
            valid_mask = (b_indices >= 0) & (b_indices < Nb)
            neg_inf = torch.finfo(local_scores.dtype).min
            local_scores = local_scores.masked_fill(~valid_mask, neg_inf)

            if self._tome_info.get("use_softkmax")==False:
                # print("use_softkmax is False")
                # print(f"local_scores: {local_scores.shape}")
                # print(f"k: {k}")
                # print(f"self._tome_info['tau2']: {self._tome_info['tau2']}")
                khot = torch.zeros_like(local_scores)
                local_scores_iter = local_scores.clone()  # Clone to avoid inplace modification
                for _ in range(k):
                    onehot_approx = F.softmax(local_scores_iter / (self._tome_info.get("tau2", 0.1)), dim=-1)
                    khot += onehot_approx
                    khot_mask = torch.clamp(1 - onehot_approx.sum(dim=-1, keepdim=True), min=EPSILON)
                    local_scores_iter = local_scores_iter + torch.log(khot_mask)
            else:
                # print("use_softkmax is True")
                # print(f"local_scores: {local_scores.shape}")
                # print(f"k: {k}")
                # print(f"self._tome_info['tau2']: {self._tome_info['tau2']}")
                local_scores_energy = torch.logsumexp(local_scores, dim=-1).exp()
                row_khot = ThreTopK(local_scores_energy, k, self._tome_info["tau2"])
                khot = F.softmax(local_scores, dim=-1) * row_khot.unsqueeze(-1)

            tmp = torch.clamp(khot.sum(dim=-1, keepdim=True).detach() - 1, min=0.) + 1.
            nkhot = khot / tmp

            # Keep assign as narrow matrix (B, Na, 2*window_size+1) for local window
            # Store b_indices to map window positions to actual b token indices
            a_indices = torch.arange(Na, device=device).view(1, -1, 1)
            window_offsets = torch.arange(-window_size, window_size + 1, device=device).view(1, 1, -1)
            b_indices = a_indices + window_offsets  # (1, Na, 2*window_size+1)
            valid_mask = (b_indices >= 0) & (b_indices < Nb)  # (1, Na, 2*window_size+1)
            
            # assign stays as narrow matrix with masked values
            assign = torch.where(valid_mask.expand(B, -1, -1), nkhot, 0.0)  # (B, Na, 2*window_size+1)
            
            # Store b_indices for later use in merging
            self._tome_info['assign_b_indices'] = b_indices  # (1, Na, 2*window_size+1)
            self._tome_info['assign_valid_mask'] = valid_mask  # (1, Na, 2*window_size+1)
            
            # Store original indices for physical locality check
            self._tome_info['assign_a_orig_idx'] = a_orig_idx  # (B, Na) or None
            self._tome_info['assign_b_orig_idx'] = b_orig_idx  # (B, Nb) or None
            # import pdb;pdb.set_trace()
        with torch.no_grad():
            out_dict = {
                'num': (nkhot if 'nkhot' in locals() else khot).sum().item(),
                'max': (khot if 'khot' in locals() else nkhot).view(B, -1).max(dim=-1)[0].sum().item(),
            }
        return assign, out_dict

    # @torch.no_grad()
    # def _build_token_map_hard(self, metric_a, metric_b, r, n, t_orig, class_token, distill_token):
    #     B, Na, D = metric_a.shape
    #     _, Nb, _ = metric_b.shape
    #     device = metric_a.device

    #     if r == 0:
    #         token_map = torch.arange(t_orig, device=device).unsqueeze(0).expand(B, -1)
    #         src_mask = torch.zeros((B, t_orig), device=device, dtype=torch.bool)
    #         return token_map, src_mask

    #     window_size = self._tome_info.get("window_size")
    #     if window_size is not None and window_size > 0:
    #         padded_b = F.pad(metric_b, (0, 0, window_size, window_size if Na == Nb else window_size + 1))
    #         b_windows_unfolded = padded_b.unfold(dimension=1, size=2 * window_size + 1, step=1)
    #         b_windows = b_windows_unfolded.permute(0, 1, 3, 2)
    #         a_reshaped = metric_a.unsqueeze(2)
    #         local_scores = (a_reshaped * b_windows).sum(dim=-1)
    #         node_max, local_node_idx = local_scores.max(dim=-1)
    #         node_idx = torch.arange(Na, device=device).view(1, -1) + local_node_idx - window_size
    #     else:
    #         scores = metric_a @ metric_b.transpose(-1, -2)
    #         node_max, node_idx = scores.max(dim=-1)
    #     edge_idx = node_max.argsort(dim=-1, descending=True)
    #     src_a_idx = edge_idx[:, :r]
    #     unm_a_idx = edge_idx[:, r:]
    #     dst_b_idx = torch.gather(node_idx, dim=1, index=src_a_idx)
    #     src_orig_idx = 1 + 2 * src_a_idx
    #     unm_orig_idx = 1 + 2 * unm_a_idx
    #     dst_orig_idx = 2 + 2 * dst_b_idx
    #     b_all_orig_idx = torch.arange(2, 2 * Nb + 2, 2, device=device).unsqueeze(0).expand(B, -1)

    #     num_unm = Na - r
    #     unm_new_idx = torch.arange(1, num_unm + 1, device=device).unsqueeze(0).expand(B, -1)
    #     b_all_new_idx = torch.arange(num_unm + 1, num_unm + 1 + Nb, device=device).unsqueeze(0).expand(B, -1)
    #     dst_new_idx = torch.gather(b_all_new_idx, dim=1, index=dst_b_idx)

    #     token_map_batch = torch.arange(t_orig, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    #     src_mask_batch = torch.zeros((B, t_orig), device=device, dtype=torch.bool)
    #     map_builder = token_map_batch.clone()
    #     if unm_orig_idx.numel() > 0:
    #         map_builder.scatter_(dim=1, index=unm_orig_idx, src=unm_new_idx)
    #     map_builder.scatter_(dim=1, index=b_all_orig_idx, src=b_all_new_idx)
    #     map_builder.scatter_(dim=1, index=src_orig_idx, src=dst_new_idx)
    #     src_mask_batch.scatter_(dim=1, index=src_orig_idx, value=True)
    #     return map_builder, src_mask_batch

    def _merge_train(self, x, size, r, n, metric, source_matrix=None):
        # print("test_merge_train")
        # print(f"r: {r}")
        # print(f"n: {n}")
        # print(self._tome_info)
        metric = metric['metric']
        metric = metric / metric.norm(dim=-1, keepdim=True)

        n = n if self.training else x.size()[1]
        r = min(r, (n - 1) // 2)
        if r <= 0:
            return x, size, n, metric, source_matrix

        # 保证 size 为 [B, N, 1]
        if size is not None and size.ndim == 2:
            size = size.unsqueeze(-1)
        
        # Initialize source_matrix if needed
        if source_matrix is None:
            B, N = x.shape[0], x.shape[1]
            window_size = self._tome_info.get("window_size", 0)
            local_depth = self._tome_info.get("local_depth", 1)
            width = 2 * window_size * local_depth + 1
            center = window_size * local_depth
            
            source_matrix = torch.zeros(B, N, width, device=x.device, dtype=x.dtype)
            source_matrix[:, :, center] = 1.0  # Identity: each token 100% from itself
            # Store metadata only, not the matrix itself
            self._tome_info["source_matrix_center"] = center
            self._tome_info["source_matrix_width"] = width

        # xa, xb = x[..., 1:n:2, :], x[..., 2:n:2, :]
        # a, b = metric[..., 1:n:2, :], metric[..., 2:n:2, :]
        # wa = size[..., 1:n:2, 0]
        # wb = size[..., 2:n:2, 0]

        # 改为随机分组
        # =================================================================================== #
        B, T, C = x.shape
        device = x.device
        
        # Random grouping: split tokens (excluding cls) into two groups
        n_tokens = n - 1  # exclude cls token at position 0
        Na = n_tokens // 2
        Nb = n_tokens - Na
        
        # Generate random permutation for each batch (parallelized)
        # Use argsort of random values to get permutation
        rand_vals = torch.rand(B, n_tokens, device=device)
        rand_perm = torch.argsort(rand_vals, dim=1)  # [B, n_tokens]
        a_idx = rand_perm[:, :Na] + 1  # +1 because we skip cls at position 0
        b_idx = rand_perm[:, Na:] + 1

        # 暂时把 a_idx 和 b_idx 改回奇偶分组
        # a_idx = torch.arange(1, n, 2, device=device).unsqueeze(0).expand(B, -1)
        # b_idx = torch.arange(2, n, 2, device=device).unsqueeze(0).expand(B, -1)
        # import pdb;pdb.set_trace()
        
        # Sort indices within each group for efficient gathering
        a_idx_sorted, a_sort_order = torch.sort(a_idx, dim=1)
        b_idx_sorted, b_sort_order = torch.sort(b_idx, dim=1)
        
        # Extract tokens using sorted indices
        # Note: x and metric have different feature dimensions
        D_metric = metric.shape[-1]
        
        a_idx_expanded_x = a_idx_sorted.unsqueeze(-1).expand(-1, -1, C)
        b_idx_expanded_x = b_idx_sorted.unsqueeze(-1).expand(-1, -1, C)
        a_idx_expanded_metric = a_idx_sorted.unsqueeze(-1).expand(-1, -1, D_metric)
        b_idx_expanded_metric = b_idx_sorted.unsqueeze(-1).expand(-1, -1, D_metric)

        # import pdb;pdb.set_trace()

        
        xa = torch.gather(x, dim=1, index=a_idx_expanded_x)
        xb = torch.gather(x, dim=1, index=b_idx_expanded_x)
        a = torch.gather(metric, dim=1, index=a_idx_expanded_metric)
        b = torch.gather(metric, dim=1, index=b_idx_expanded_metric)
        wa = torch.gather(size[..., 0], dim=1, index=a_idx_sorted)
        wb = torch.gather(size[..., 0], dim=1, index=b_idx_sorted)
        
        # Extract source matrix for a and b tokens
        if source_matrix is not None:
            width = self._tome_info["source_matrix_width"]
            a_idx_expanded_source = a_idx_sorted.unsqueeze(-1).expand(-1, -1, width)
            b_idx_expanded_source = b_idx_sorted.unsqueeze(-1).expand(-1, -1, width)
            source_a = torch.gather(source_matrix, dim=1, index=a_idx_expanded_source)
            source_b = torch.gather(source_matrix, dim=1, index=b_idx_expanded_source)
            
            # Save old weights for source update
            wa_old = wa.clone()
            wb_old = wb.clone()

        # =================================================================================== #

        # 对齐合并粒度 t
        t = max(int(self._tome_info.get("t", 1)), 1)
        r = (r // t) * t
        # print(f"r: {r}")
        # print(f"a: {a.shape}")
        # print(f"b: {b.shape}")
        assign, _out = self._select(k=r, a=a, b=b, 
                                    a_orig_idx=a_idx_sorted, 
                                    b_orig_idx=b_idx_sorted)

        # Handle narrow matrix case for local window
        window_size = self._tome_info.get("window_size")
        if window_size is not None and window_size > 0:
            # assign is (B, Na, 2*window_size+1), need to scatter to (B, Nb, Na)
            B_cur, Na_cur = assign.shape[0], assign.shape[1]
            Nb_cur = xb.shape[1]
            C_cur = xa.shape[-1]
            
            # Get stored b_indices and valid_mask
            b_indices = self._tome_info['assign_b_indices']  # (1, Na, 2*window_size+1)
            valid_mask = self._tome_info['assign_valid_mask']  # (1, Na, 2*window_size+1)
            
            # Get original indices for physical locality check
            a_orig_idx = self._tome_info.get('assign_a_orig_idx')  # (B, Na) or None
            b_orig_idx = self._tome_info.get('assign_b_orig_idx')  # (B, Nb) or None
            
            # Compute physical locality mask if original indices are provided
            physical_mask = valid_mask.clone()  # Start with valid_mask
            if a_orig_idx is not None and b_orig_idx is not None:
                # For each a[i], check if |a_orig_idx[i] - b_orig_idx[b_indices[i,j]]| <= window_size
                # a_orig_idx: (B, Na) -> (B, Na, 1)
                # b_indices: (1, Na, 2*window_size+1)
                # b_orig_idx: (B, Nb)
                
                a_orig_expanded = a_orig_idx.unsqueeze(2)  # (B, Na, 1)
                b_indices_clamped = b_indices.clamp(0, Nb_cur - 1)  # (1, Na, 2*window_size+1)
                b_indices_expanded = b_indices_clamped.expand(B_cur, -1, -1)  # (B, Na, 2*window_size+1)
                
                # Gather b's original positions at window indices
                # b_orig_idx: (B, Nb) -> expand to (B, Na, Nb) -> gather with index (B, Na, 2*w+1)
                b_orig_expanded_for_gather = b_orig_idx.unsqueeze(1).expand(-1, Na_cur, -1)  # (B, Na, Nb)
                b_orig_at_window = torch.gather(b_orig_expanded_for_gather, dim=2, 
                                               index=b_indices_expanded)  # (B, Na, 2*w+1)
                
                # Check physical distance
                physical_distance = torch.abs(a_orig_expanded - b_orig_at_window)  # (B, Na, 2*w+1)
                physical_local_mask = physical_distance <= window_size  # (B, Na, 2*w+1)
                
                # Combine with valid_mask
                physical_mask = valid_mask & physical_local_mask  # (B, Na, 2*w+1)
            
            # Apply physical mask to assign
            assign = assign * physical_mask.float()
            
            # Clamp indices for scatter operation
            b_indices_clamped = b_indices.clamp(0, Nb_cur - 1)  # (1, Na, 2*window_size+1)
            
            # Compute weighted xa: (B, Na, C)
            weighted_xa = wa[..., None] * xa  # (B, Na, C)
            
            # For each b token, accumulate from corresponding a tokens in window
            # We need to scatter assign.transpose(-1, -2) @ weighted_xa
            # assign: (B, Na, 2*window_size+1)
            # weighted_xa: (B, Na, C)
            
            # Expand assign for broadcasting with xa
            assign_expanded = assign.unsqueeze(-1)  # (B, Na, 2*window_size+1, 1)
            weighted_xa_expanded = weighted_xa.unsqueeze(2)  # (B, Na, 1, C)
            contribution = assign_expanded * weighted_xa_expanded  # (B, Na, 2*window_size+1, C)
            
            # Scatter contributions to xb
            # For scatter_add on dim=1, index shape must match src shape
            b_indices_expanded = b_indices_clamped.expand(B_cur, -1, -1).unsqueeze(-1).expand(-1, -1, -1, C_cur)  # (B, Na, 2*window_size+1, C)
            xb_contrib = torch.zeros(B_cur, Nb_cur, C_cur, device=xb.device, dtype=xb.dtype)
            
            # Reshape for scatter: (B, Na*window, C)
            contribution_flat = contribution.reshape(B_cur, -1, C_cur)  # (B, Na*(2*window_size+1), C)
            b_indices_flat = b_indices_expanded.reshape(B_cur, -1, C_cur)  # (B, Na*(2*window_size+1), C)
            xb_contrib.scatter_add_(dim=1, index=b_indices_flat, src=contribution_flat)
            xb = wb[..., None] * xb + xb_contrib
            
            # Similar for wb
            wa_expanded = wa.unsqueeze(2)  # (B, Na, 1)
            wb_contribution = assign_expanded[..., 0] * wa_expanded  # (B, Na, 2*window_size+1)
            b_indices_expanded_1d = b_indices_clamped.expand(B_cur, -1, -1)  # (B, Na, 2*window_size+1)
            wb_contrib = torch.zeros(B_cur, Nb_cur, device=wb.device, dtype=wb.dtype)
            
            # Reshape for scatter: (B, Na*window)
            wb_contribution_flat = wb_contribution.reshape(B_cur, -1)  # (B, Na*(2*window_size+1))
            b_indices_flat_1d = b_indices_expanded_1d.reshape(B_cur, -1)  # (B, Na*(2*window_size+1))
            wb_contrib.scatter_add_(dim=1, index=b_indices_flat_1d, src=wb_contribution_flat)
            wb = wb + wb_contrib
            
            # Compute tmp: 1 - sum of assign over last dim
            tmp = 1 - assign.sum(dim=-1)  # (B, Na)
        else:
            # Global case: assign is (B, Na, Nb)
            xb = wb[..., None] * xb + assign.transpose(-1, -2) @ (wa[..., None] * xa)
            wb = wb + (assign.transpose(-1, -2) @ wa[..., None])[..., 0]
            tmp = 1 - assign.sum(dim=-1)
        
        wa = wa * (tmp + (torch.clamp(tmp, min=0., max=1.) - tmp).detach())
        wb_safe = torch.clamp(wb, min=torch.finfo(wb.dtype).eps)
        xb = xb / wb_safe[..., None]
        
        # Update source_matrix
        if source_matrix is not None:
            width = self._tome_info["source_matrix_width"]
            center = self._tome_info["source_matrix_center"]
            
            window_size_check = self._tome_info.get("window_size")
            
            if window_size_check is not None and window_size_check > 0:
                # Local window case: update source_matrix based on assign contributions
                
                # Prepare increment for source_b
                source_increment = torch.zeros_like(source_b)  # (B, Nb, width)
                
                # Get stored indices
                b_indices = self._tome_info['assign_b_indices']  # (1, Na, 2*window_size+1)
                b_indices_clamped = b_indices.clamp(0, Nb_cur - 1).expand(B_cur, -1, -1)  # (B, Na, 2*w+1)
                
                # Compute delta: physical distance from a to b
                a_orig_expanded = a_idx_sorted.unsqueeze(2)  # (B, Na, 1)
                b_orig_at_window = torch.gather(
                    b_idx_sorted.unsqueeze(1).expand(-1, Na_cur, -1),
                    dim=2, index=b_indices_clamped
                )  # (B, Na, 2*w+1)
                delta = a_orig_expanded - b_orig_at_window  # (B, Na, 2*w+1)
                
                # Update source_matrix with correct logic:
                # When a[i] contributes weight w to b[j]:
                # 1. b[j] receives a[i]'s all sources scaled by w (with position shift)
                # 2. a[i] loses those contributions (scaled by w)
                
                # Prepare: source_b starts with old values
                source_b_new = source_b.clone()
                source_a_new = source_a.clone()
                
                # Fully vectorized update
                # assign: (B, Na, 2*w+1), represents a[i] -> b[j] with weight assign[i, j_local]
                # source_a: (B, Na, width)
                # delta: (B, Na, 2*w+1), physical distance from a[i] to b[j]
                
                # For each a[i], it transfers assign[i, j_local] to corresponding b[j]
                # source_a[i, k] represents a[i]'s contribution to relative position k-center
                # After transfer, b[j] contributes to the same physical position
                # Relative to b[j], the offset becomes k + delta[i, j_local]
                
                # Expand dimensions for vectorization
                # source_a: (B, Na, width) -> (B, Na, 1, width)
                # assign: (B, Na, 2*w+1) -> (B, Na, 2*w+1, 1)
                source_a_4d = source_a.unsqueeze(2)  # (B, Na, 1, width)
                assign_4d = assign.unsqueeze(-1)  # (B, Na, 2*w+1, 1)
                
                # Transferred contributions: source_a[i, k] * assign[i, j_local]
                transferred = source_a_4d * assign_4d  # (B, Na, 2*w+1, width)
                
                # Compute target positions in b's reference frame
                # k_range: (1, 1, 1, width)
                # delta: (B, Na, 2*w+1, 1)
                k_range = torch.arange(width, device=x.device).view(1, 1, 1, -1)
                delta_4d = delta.unsqueeze(-1)  # (B, Na, 2*w+1, 1)
                
                # New position in b's frame: k + delta
                target_k = k_range + delta_4d  # (B, Na, 2*w+1, width)
                valid_target = (target_k >= 0) & (target_k < width)
                target_k_clamped = target_k.clamp(0, width - 1)
                
                # Apply valid mask
                transferred_masked = transferred * valid_target.float()
                
                # Scatter to source_b using b_indices
                # b_indices_clamped: (B, Na, 2*w+1)
                # We need to add transferred_masked[b, i, j_local, k] to source_b_new[b, j, target_k]
                
                batch_idx = torch.arange(B_cur, device=x.device).view(-1, 1, 1, 1)
                b_idx_4d = b_indices_clamped.unsqueeze(-1)  # (B, Na, 2*w+1, 1)
                
                # Linear index: batch * (Nb * width) + b_idx * width + target_k
                linear_idx = batch_idx * Nb_cur * width + b_idx_4d * width + target_k_clamped
                
                # Flatten and scatter_add
                linear_idx_flat = linear_idx.reshape(-1)
                transferred_flat = transferred_masked.reshape(-1)
                
                source_b_flat = source_b_new.reshape(-1)
                source_b_flat.scatter_add_(0, linear_idx_flat, transferred_flat)
                source_b_new = source_b_flat.reshape(B_cur, Nb_cur, width)
                
                # Update source_a: reduce by total transferred amount
                # Total transferred from a[i]: sum over j_local of (assign[i, j_local] * source_a[i, :])
                total_transferred = (assign.unsqueeze(-1) * source_a.unsqueeze(2)).sum(dim=2)  # (B, Na, width)
                source_a_new = source_a - total_transferred
            else:
                # Global case: similar logic but with full assign matrix
                # For simplicity, keep source unchanged in global mode for now
                source_a_new = source_a
                source_b_new = source_b
            
            # Write back to full source_matrix
            source_matrix_new = source_matrix.clone()
            
            # Scatter a's sources back
            a_idx_expanded_source = a_idx_sorted.unsqueeze(-1).expand(-1, -1, width)
            source_matrix_new.scatter_(dim=1, index=a_idx_expanded_source, src=source_a_new)
            
            # Scatter b's sources back
            b_idx_expanded_source = b_idx_sorted.unsqueeze(-1).expand(-1, -1, width)
            source_matrix_new.scatter_(dim=1, index=b_idx_expanded_source, src=source_b_new)
            
            # Optional: normalize to ensure sum = 1 (compensate for floating point errors)
            if self._tome_info.get("normalize_source_matrix", False):
                source_sum = source_matrix_new.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                source_matrix_new = source_matrix_new / source_sum
            
            source_matrix = source_matrix_new
        
        # =================================================================================== #
        # w = torch.cat([wa, wb], dim=-1)
        # nx = torch.cat([xa, xb], dim=1)
        # nidxs = w.argsort(dim=-1, descending=True)
        # w = w.gather(dim=-1, index=nidxs)
        # nx = nx.gather(dim=-2, index=nidxs[..., None].expand_as(nx))
        # =================================================================================== #

        # =================================================================================== #
        
        # Restore original spatial order based on the random grouping
        # SOFT MERGE: Keep ALL tokens (both a and b), only weights change
        # wa becomes smaller for merged tokens, wb becomes larger for receiving tokens
        Na_curr = xa.shape[1]  # Should equal Na (all a tokens kept)
        Nb_curr = xb.shape[1]  # Should equal Nb (all b tokens kept)
        n_total = Na_curr + Nb_curr  # Should equal Na + Nb
        
        # Combine ALL a tokens and ALL b tokens with their original positions
        all_orig_pos = torch.cat([a_idx_sorted, b_idx_sorted], dim=1)  # [B, Na+Nb]
        all_tokens = torch.cat([xa, xb], dim=1)  # [B, Na+Nb, C]
        all_weights = torch.cat([wa, wb], dim=1)  # [B, Na+Nb]
        
        # Sort by original position to restore spatial order
        sort_indices = torch.argsort(all_orig_pos, dim=1)  # [B, Na+Nb]
        sort_indices_expanded = sort_indices.unsqueeze(-1).expand(-1, -1, C)
        
        nx = torch.gather(all_tokens, dim=1, index=sort_indices_expanded)
        w = torch.gather(all_weights, dim=1, index=sort_indices)

        # =================================================================================== #
        
        # import pdb;pdb.set_trace()

        x_output = torch.cat([x[:, :1], nx, x[:, n:]], dim=1)
        size_output = torch.cat([size[:, :1, 0], w, size[:, n:, 0]], dim=-1).unsqueeze(-1)
        return x_output, size_output, n - r, _out, source_matrix

    def _merge_eval(self, x, size, r, metric, source_matrix=None):    # eval：保留 ToMe 行为。弃用。
        metric = metric['metric']
        metric = metric / metric.norm(dim=-1, keepdim=True)

        merge, _, current_level_map = bipartite_soft_matching(metric,
                                           r,
                                           self._tome_info["class_token"],
                                           self._tome_info["distill_token"],
                                           )
        if self._tome_info["trace_source"]:
            # 兼容旧 map/matrix 追踪
            if "source_tracking_mode" in self._tome_info:
                if self._tome_info["source_tracking_mode"] == 'map':
                    source_map = self._tome_info["source_map"]
                    if source_map is None:
                        b, t, _ = x.shape
                        source_map = torch.arange(t, device=x.device, dtype=torch.long).expand(b, -1)
                    self._tome_info["source_map"] = merge_source_map(current_level_map, x, source_map)
                else: # 'matrix'
                    source_matrix = self._tome_info["source_matrix"]
                    self._tome_info["source_matrix"] = merge_source_matrix(merge, x, source_matrix)

        x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
        return x, self._tome_info["size"], x.size(1), None, source_matrix

    def merge(self, x, size, r, n, metric, source_matrix=None):
        # import pdb;pdb.set_trace()
        if not self._tome_info["use_cross_attention"]:
            # return self._merge_train(x, size, r, n, metric, source_matrix) if self.training else self._merge_eval(x, size, r, metric, source_matrix)
            raise ValueError("Cross attention is not supported")
        else:
            return self._merge_train(x, size, r, n, metric, source_matrix)

    def forward(self, x, size, n=None, source_matrix=None):
        if size is None:
            size=torch.ones_like(x[..., 0, None])
        tmp, metric = self.attn(self.norm1(x), size=size)
        assert isinstance(metric['metric'], (float, torch.Tensor)), "metric not a float or torch.Tensor"
        x = x + self.drop_path1(self.ls1(tmp))
        # Merging
        r = self._tome_info["r"].pop(0)
        if size is not None and r > 0 and n > 0:
            x, size, n, metric, source_matrix = self.merge(x, size, r, n, metric, source_matrix)
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        return x, size, n, metric, source_matrix


def make_tome_class(transformer_class):
    class DTEMVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward_features(self, x):
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)

            n = x.size(1)
            self._tome_info["r"] = parse_r(
                len(self.blocks), self.r, self._tome_info["total_merge"]
            )
            self._tome_info["size"] = torch.ones_like(x[..., 0, None])
            self._tome_info["source_map"] = None

            out_dicts = []
            source_matrix = None  # Initialize, will be created in first block
            for block in self.blocks:
                x, size, n, out_dict, source_matrix = block(x, self._tome_info["size"], n=n, source_matrix=source_matrix)
                out_dicts.append(out_dict)

            x = self.norm(x)
            return x, out_dicts

        def forward(self, x, return_out_dicts=False):
            x, out_dicts = self.forward_features(x)
            x = self.forward_head(x)
            if return_out_dicts:
                return x, out_dicts
            return x

    return DTEMVisionTransformer



""""
Learning to Merge Tokens via Decoupled Embedding for Efficient Vision Transformers, NIPS'2024
    - paper (https://openreview.net/forum?id=pVPyCgXv57) 
    - code  (https://github.com/movinghoon/DTEM)
"""
def dtem_apply_patch(
    model: VisionTransformer,
    feat_dim=None,
    trace_source=True,
    prop_attn=True,
    source_tracking_mode: str = 'map',
    default_r: int = 2,
    window_size: int = None,
    t: int = 1,
    use_softkmax: bool = False,
    use_cross_attention: bool = True,
):
    """
    扩展：
    - default_r: 默认合并强度（用于初始化 model.r）
    - window_size: 局部注意力/局部匹配窗口半径（None 或 <=0 表示禁用）
    - t: 合并粒度，使每层实际 r 对齐到 t 的整数倍
    """
    DTEMVisionTransformer = make_tome_class(model.__class__)
    model.__class__ = DTEMVisionTransformer

    model.r = default_r
    model._tome_info = {
        "r": model.r,
        "size": None,
        # 旧版追踪结构：保留兼容
        "source_map": None,
        "source_matrix": None,
        "total_merge": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": getattr(model, 'cls_token', None) is not None,
        "distill_token": getattr(model, 'dist_token', None) is not None,
        "source_tracking_mode": source_tracking_mode,
        # DTEM 超参
        "k2": None,
        "tau1": 1.0,
        "tau2": 0.1,
        "feat_dim": feat_dim,
        # 局部窗口与合并粒度
        "window_size": window_size,
        "t": t,
        "use_softkmax": use_softkmax,
        "use_cross_attention": use_cross_attention,
    }
    for module in model.modules():
        if isinstance(module, (Block, TimmBlock)):
            module.__class__ = DTEMBlock
            module._tome_info = model._tome_info
        elif isinstance(module, (Attention, TimmAttention)):
            module.__class__ = DTEMAttention
            module._tome_info = model._tome_info
            module.patch(model._tome_info["feat_dim"])
