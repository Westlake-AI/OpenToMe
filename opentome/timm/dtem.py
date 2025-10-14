# ------ jinxin modified ------ #
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath, use_fused_attn
from timm.models.vision_transformer import VisionTransformer, LayerScale
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock
from functools import partial

from opentome.tome.tome import (
    bipartite_soft_matching,
    merge_source_matrix,
    merge_source_map,
    merge_wavg,
    parse_r,
)
from opentome.timm import Attention, Block


# ---------------------- dtem 专用通用 map/trace 工具 ---------------------- #
def merge_source_map_for_dtem(current_level_map, source_map):
    if source_map is None:
        return current_level_map
    return torch.gather(current_level_map, dim=1, index=source_map)

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
        # qkv -> self
        self.weight[:-self.feat_dim].copy_(self.qkv_layer.weight)
        self.bias[:-self.feat_dim].copy_(self.qkv_layer.bias)
        
        # metric_layer -> self
        self.weight[-self.feat_dim:].copy_(self.metric_layer.weight)
        self.bias[-self.feat_dim:].copy_(self.metric_layer.bias)

    def train(self, mode=True):
        if mode is False:   # if eval
            self.update()
        return super().train(mode)

    def forward(self, input: torch.Tensor):
        if not self.training:
            out = F.linear(input, self.weight, self.bias)
            return out[..., :-self.feat_dim], out[..., -self.feat_dim:]
        
        # training
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

        # fp32 for softmax computation
        q, k, v = q.type(torch.float32), k.type(torch.float32), v.type(torch.float32)
        with torch.cuda.amp.autocast(dtype=torch.float32, enabled=True):
            q = q * self.scale
            window_size = self._tome_info.get("window_size")
            if window_size is not None and window_size > 0:
                # 尝试使用 triton 实现；失败回退到 naive/flash
                x_bnc = None
                try:
                    from .local_attn_triton import local_attn_sizebias_bhnd
                    x_bnhd = local_attn_sizebias_bhnd(q, k, v, size, window_size, softmax_scale=1.0)  # (B, N, H, D)
                    x_bnc = x_bnhd.reshape(B, N, self.num_heads * self.head_dim)
                    x_bnc = self.attn_drop(x_bnc)
                except Exception:
                    # 先试 flash，再退回 naive
                    x_bnc = self._flash_local_attention(q, k, v, size, window_size, x_dtype=x.dtype) if FLASH_ATTN_AVAILABLE else None
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

    def _flash_local_attention(self, q, k, v, size, window_size, x_dtype=torch.float32):
        """
        使用 FlashAttention 的局部窗口注意力；失败返回 None。
        """
        B, H, N, D = q.shape
        try:
            q_nhw = q.permute(0, 2, 1, 3).contiguous()
            k_nhw = k.permute(0, 2, 1, 3).contiguous()
            v_nhw = v.permute(0, 2, 1, 3).contiguous()

            if q_nhw.dtype not in (torch.float16, torch.bfloat16):
                q_nhw = q_nhw.to(torch.float16)
                k_nhw = k_nhw.to(torch.float16)
                v_nhw = v_nhw.to(torch.float16)

            out = flash_attn_func(
                q_nhw, k_nhw, v_nhw,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=1.0,
                causal=False,
                window_size=(window_size, window_size),
                deterministic=True,
            )

            if out.ndim == 3 and out.shape[-1] == H * D:
                out = out.view(B, N, H, D)
            elif out.ndim != 4 or out.shape[:3] != (B, N, H):
                out = out.reshape(B, N, H, D)

            attn_out_bhnd = out.permute(0, 2, 1, 3).contiguous()
            x_bnc = attn_out_bhnd.transpose(1, 2).reshape(B, N, H * D).to(x_dtype)
            return x_bnc
        except Exception:
            return None



class DTEMBlock(Block):
    
    def _select(self, k, a, b):
        """
        统一的选择逻辑：支持全局或局部窗口匹配，并支持跨层屏蔽。
        a: (B, Na, C), b: (B, Nb, C)
        返回 assign: (B, Na, Nb)
        """
        EPSILON = torch.finfo(torch.float32).eps
        B, Na, C = a.shape
        _, Nb, _ = b.shape
        device = a.device

        window_size = self._tome_info.get("window_size")
        if window_size is None or window_size <= 0:
            scores = a @ b.transpose(-1, -2)
            token_mask = self._tome_info.get("token_mask_for_dtem", None)
            if token_mask is not None:
                Na_cur = a.shape[1]
                Nb_cur = b.shape[1]
                mask_a = token_mask[:, 1:1 + 2 * Na_cur:2]
                mask_b = token_mask[:, 2:2 + 2 * Nb_cur:2]
                neg_inf = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(mask_a.unsqueeze(-1), neg_inf)
                scores = scores.masked_fill(mask_b.unsqueeze(1), neg_inf)
            _idx = scores.argsort(dim=-1, descending=True)
            _x = scores.gather(dim=-1, index=_idx)
            _x = _x / (self._tome_info.get("tau1", 1.0))
            khot = torch.zeros_like(_x)
            for _ in range(k):
                onehot_approx = F.softmax(_x.view(B, -1) / (self._tome_info.get("tau2", 0.1)), dim=-1).view(B, Na, Nb)
                khot += onehot_approx
                khot_mask = torch.clamp(1 - onehot_approx.sum(dim=-1, keepdim=True), min=EPSILON)
                _x = _x + torch.log(khot_mask)
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

            token_mask = self._tome_info.get("token_mask_for_dtem", None)
            if token_mask is not None:
                Na_cur = a.shape[1]
                Nb_cur = b.shape[1]
                mask_a = token_mask[:, 1:1 + 2 * Na_cur:2]
                mask_b = token_mask[:, 2:2 + 2 * Nb_cur:2]
                neg_inf_local = torch.finfo(local_scores.dtype).min
                local_scores = local_scores.masked_fill(mask_a.unsqueeze(-1), neg_inf_local)
                padded_mask_b = F.pad(mask_b, (window_size, window_size))
                mask_windows_unfolded = padded_mask_b.unfold(dimension=1, size=2*window_size+1, step=1)
                local_scores = local_scores.masked_fill(mask_windows_unfolded, neg_inf_local)

            khot = torch.zeros_like(local_scores)
            for _ in range(k):
                onehot_approx = F.softmax(local_scores / (self._tome_info.get("tau2", 0.1)), dim=-1)
                khot += onehot_approx
                khot_mask = torch.clamp(1 - onehot_approx.sum(dim=-1, keepdim=True), min=EPSILON)
                local_scores = local_scores + torch.log(khot_mask)

            tmp = torch.clamp(khot.sum(dim=-1, keepdim=True).detach() - 1, min=0.) + 1.
            nkhot = khot / tmp

            a_indices = torch.arange(Na, device=device).view(1, -1, 1)
            window_offsets = torch.arange(-window_size, window_size + 1, device=device).view(1, 1, -1)
            b_indices = a_indices + window_offsets
            valid_mask = (b_indices >= 0) & (b_indices < Nb)
            assign = torch.zeros(B, Na, Nb, device=device, dtype=a.dtype)
            nkhot_masked = torch.where(valid_mask.expand(B, -1, -1), nkhot, 0.0)
            b_indices_clamped = b_indices.clamp(0, Nb - 1)
            b_indices_expanded = b_indices_clamped.expand(B, -1, -1)
            assign.scatter_add_(dim=2, index=b_indices_expanded, src=nkhot_masked)

        with torch.no_grad():
            out_dict = {
                'num': (nkhot if 'nkhot' in locals() else khot).sum().item(),
                'max': (khot if 'khot' in locals() else nkhot).view(B, -1).max(dim=-1)[0].sum().item(),
            }
        return assign, out_dict

    @torch.no_grad()
    def _build_token_map_hard(self, metric_a, metric_b, r, n, t_orig, class_token, distill_token):
        B, Na, D = metric_a.shape
        _, Nb, _ = metric_b.shape
        device = metric_a.device

        if r == 0:
            token_map = torch.arange(t_orig, device=device).unsqueeze(0).expand(B, -1)
            src_mask = torch.zeros((B, t_orig), device=device, dtype=torch.bool)
            return token_map, src_mask

        window_size = self._tome_info.get("window_size")
        if window_size is not None and window_size > 0:
            padded_b = F.pad(metric_b, (0, 0, window_size, window_size if Na == Nb else window_size + 1))
            b_windows_unfolded = padded_b.unfold(dimension=1, size=2 * window_size + 1, step=1)
            b_windows = b_windows_unfolded.permute(0, 1, 3, 2)
            a_reshaped = metric_a.unsqueeze(2)
            local_scores = (a_reshaped * b_windows).sum(dim=-1)
            node_max, local_node_idx = local_scores.max(dim=-1)
            node_idx = torch.arange(Na, device=device).view(1, -1) + local_node_idx - window_size
        else:
            scores = metric_a @ metric_b.transpose(-1, -2)
            node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)
        src_a_idx = edge_idx[:, :r]
        unm_a_idx = edge_idx[:, r:]
        dst_b_idx = torch.gather(node_idx, dim=1, index=src_a_idx)
        src_orig_idx = 1 + 2 * src_a_idx
        unm_orig_idx = 1 + 2 * unm_a_idx
        dst_orig_idx = 2 + 2 * dst_b_idx
        b_all_orig_idx = torch.arange(2, 2 * Nb + 2, 2, device=device).unsqueeze(0).expand(B, -1)

        num_unm = Na - r
        unm_new_idx = torch.arange(1, num_unm + 1, device=device).unsqueeze(0).expand(B, -1)
        b_all_new_idx = torch.arange(num_unm + 1, num_unm + 1 + Nb, device=device).unsqueeze(0).expand(B, -1)
        dst_new_idx = torch.gather(b_all_new_idx, dim=1, index=dst_b_idx)

        token_map_batch = torch.arange(t_orig, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        src_mask_batch = torch.zeros((B, t_orig), device=device, dtype=torch.bool)
        map_builder = token_map_batch.clone()
        if unm_orig_idx.numel() > 0:
            map_builder.scatter_(dim=1, index=unm_orig_idx, src=unm_new_idx)
        map_builder.scatter_(dim=1, index=b_all_orig_idx, src=b_all_new_idx)
        map_builder.scatter_(dim=1, index=src_orig_idx, src=dst_new_idx)
        src_mask_batch.scatter_(dim=1, index=src_orig_idx, value=True)
        return map_builder, src_mask_batch

    def _merge_train(self, x, size, r, n, metric):
        metric = metric['metric']
        metric = metric / metric.norm(dim=-1, keepdim=True)

        n = n if self.training else x.size()[1]
        r = min(r, (n - 1) // 2)
        if r <= 0:
            return x, size, n, metric

        # 保证 size 为 [B, N, 1]
        if size is not None and size.ndim == 2:
            size = size.unsqueeze(-1)

        xa, xb = x[..., 1:n:2, :], x[..., 2:n:2, :]
        a, b = metric[..., 1:n:2, :], metric[..., 2:n:2, :]
        wa = size[..., 1:n:2, 0]
        wb = size[..., 2:n:2, 0]

        # 对齐合并粒度 t
        t = max(int(self._tome_info.get("t", 1)), 1)
        r = (r // t) * t
        assign, _out = self._select(k=r, a=a, b=b)

        # 层间屏蔽保护（如果存在）
        token_mask = self._tome_info.get("token_mask_for_dtem", None)
        if token_mask is not None:
            mask_a = token_mask[:, 1:n:2]
            mask_b = token_mask[:, 2:n:2]
            combined_mask = mask_a.unsqueeze(2) | mask_b.unsqueeze(1)
            assign.masked_fill_(combined_mask, 0)

        with torch.no_grad():
            B = a.shape[0]
            T_full = x.shape[1]
            current_level_map, src_mask_batch = self._build_token_map_hard(
                a, b, r, B, T_full,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info.get("token_map_for_dtem", None) is None:
                self._tome_info["token_map_for_dtem"] = torch.arange(T_full, device=x.device)[None, :].expand(B, -1).clone()
            if self._tome_info.get("token_mask_for_dtem", None) is None:
                self._tome_info["token_mask_for_dtem"] = torch.zeros(B, T_full, dtype=torch.bool, device=x.device)
            token_map_global = merge_source_map_for_dtem(current_level_map, self._tome_info["token_map_for_dtem"])
            self._tome_info["token_map_for_dtem"] = token_map_global
            self._tome_info["token_mask_for_dtem"] = self._tome_info["token_mask_for_dtem"] | src_mask_batch

        xb = wb[..., None] * xb + assign.transpose(-1, -2) @ (wa[..., None] * xa)
        wb = wb + (assign.transpose(-1, -2) @ wa[..., None])[..., 0]
        tmp = 1 - assign.sum(dim=-1)
        wa = wa * (tmp + (torch.clamp(tmp, min=0., max=1.) - tmp).detach())
        wb_safe = torch.clamp(wb, min=torch.finfo(wb.dtype).eps)
        xb = xb / wb_safe[..., None]

        w = torch.cat([wa, wb], dim=-1)
        nx = torch.cat([xa, xb], dim=1)
        nidxs = w.argsort(dim=-1, descending=True)
        w = w.gather(dim=-1, index=nidxs)
        nx = nx.gather(dim=-2, index=nidxs[..., None].expand_as(nx))

        x_output = torch.cat([x[:, :1], nx, x[:, n:]], dim=1)
        size_output = torch.cat([size[:, :1, 0], w, size[:, n:, 0]], dim=-1).unsqueeze(-1)
        return x_output, size_output, n - r, _out

    def _merge_eval(self, x, size, r, metric):    # eval：保留 ToMe 行为并补充 token_map_for_dtem 追踪
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
            # 新 token_map_for_dtem 追踪（总是尝试维护）
            token_map = self._tome_info.get("token_map_for_dtem", None)
            b, t, _ = x.shape
            if token_map is None:
                token_map = torch.arange(t, device=x.device, dtype=torch.long).expand(b, -1)
            self._tome_info["token_map_for_dtem"] = merge_source_map_for_dtem(current_level_map, token_map)

        x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
        return x, self._tome_info["size"], x.size(1), None

    def merge(self, x, size, r, n, metric):
        return self._merge_train(x, size, r, n, metric) if self.training else self._merge_eval(x, size, r, metric)

    def forward(self, x, size, n=None):
        if size is None or n is None:
            tmp, _ = self.attn(self.norm1(x))
            x = x + self.drop_path1(self.ls1(tmp))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x
        else:
            tmp, metric = self.attn(self.norm1(x), size=size)
            assert isinstance(metric['metric'], (float, torch.Tensor)), "metric not a float or torch.Tensor"
            x = x + self.drop_path1(self.ls1(tmp))
            # Merging
            r = self._tome_info["r"].pop(0)
            if size is not None and r > 0 and n > 0:
                x, size, n, metric = self.merge(x, size, r, n, metric)
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            
            return x, size, n, metric


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
            self._tome_info["source_matrix"] = None

            out_dicts = []
            for block in self.blocks:
                x, size, n, out_dict = block(x, self._tome_info["size"], n=n)
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
        # 新的持续 token 追踪与屏蔽
        "token_map_for_dtem": None,   # [B, T]
        "token_mask_for_dtem": None,  # [B, T] bool
        # 局部窗口与合并粒度
        "window_size": window_size,
        "t": t,
    }

    for module in model.modules():
        if isinstance(module, (Block, TimmBlock)):
            module.__class__ = DTEMBlock
            module._tome_info = model._tome_info
        elif isinstance(module, (Attention, TimmAttention)):
            module.__class__ = DTEMAttention
            module._tome_info = model._tome_info
            module.patch(model._tome_info["feat_dim"])
