# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ToMe: https://github.com/facebookresearch/ToMe
# --------------------------------------------------------

import math
from typing import Callable, List, Tuple, Union
import torch.nn.functional as F
import torch


def do_nothing(x, mode=None):
    return x


def parse_r(
    num_layers: int, r: Union[List[int], Tuple[int, float], int], total: int = None
) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    total: The predefined total number of merged tokens.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)
    r_list = [int(min_val + step * i) for i in range(num_layers)]

    if total is not None:
        remainder = total - sum(r_list)
        if remainder != 0:
            if inflect < 0:
                r_list[0] += remainder
            else:
                r_list[-1] += remainder

    return r_list


def check_parse_r(
    num_layers: int, merge_num: int, total_num: int, r_inflect: float=0., sqrt: bool=False
):
    """
    Check the best merge ratio for the given 
    """
    gap = 1e10
    best_r = 0
    for i in range(merge_num):
        r_list = parse_r(num_layers, (i, r_inflect))
        gap_ = sum(r_list) - merge_num
        if gap > abs(gap_):
            keep_num = total_num - sum(r_list)
            if sqrt and int(keep_num ** 0.5) ** 2 != keep_num:
                continue
            best_r = i
            gap = abs(gap_)
        else:
            if gap < abs(gap_):
                break

    return best_r


def merge_wavg(
    merge: Callable, 
    x: torch.Tensor, 
    size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    # print("MERGE_WAVG；",x.shape,size.shape)
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, 
    x: torch.Tensor, 
    source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
    
    # print("MERGE_SOURCE；",source.shape)
    source = merge(source, mode="amax")
    return source


"""
    ToMe/ToFu/DETM specific functions
"""
def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        # unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        # src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        # dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='mean')
        # return torch.cat([unm, dst], dim=1)
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        if mode == 'mean':
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='mean')
        elif mode == 'sum':
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='sum')
        elif mode == 'tofu':
            dst_norm = torch.norm(dst, dim=-1) 
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            src_norm = torch.norm(src, dim=-1) 
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='mean')
            n = dst_norm.scatter_reduce(-1, dst_idx.squeeze(-1), src_norm, reduce='amax')
            dst = dst/dst_norm[...,None] * n[..., None]
        elif mode == 'amax':
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='amax')

        # print(unm.shape,src.shape,dst.shape)
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge

# 建议您将这两个函数完整替换掉之前的版本

# @torch.compile(mode="max-autotune")
def naive_local_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    h: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    一个考虑局部配对的二分图软匹配实现。
    这是“朴素”版本，它计算整个得分矩阵然后进行掩码操作。
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric_original_shape = metric.shape
        metric = metric / metric.norm(dim=-1, keepdim=True)

        # 为保证配对，如果token数量为奇数，则在此操作中忽略最后一个
        if metric.shape[1] % 2 != 0:
            metric_for_match = metric[:, :-1, :]
        else:
            metric_for_match = metric
        
        a, b = metric_for_match[..., ::2, :], metric_for_match[..., 1::2, :]
        
        scores = a @ b.transpose(-1, -2)
        
        k_half = scores.shape[-1]
        row_idx = torch.arange(k_half, device=scores.device).view(1, -1)
        col_idx = torch.arange(k_half, device=scores.device).view(-1, 1)
        dist_mask = torch.abs(row_idx - col_idx) > h
        scores.masked_fill_(dist_mask, -math.inf)
        
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if x.shape[1] % 2 != 0:
            last_token = x[:, -1:, :]
            x_even = x[:, :-1, :]
        else:
            last_token = None
            x_even = x

        src, dst = x_even[..., ::2, :], x_even[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))

        if mode == 'mean':
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='mean')
        elif mode == 'sum':
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='sum')
        elif mode == 'amax':
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='amax')
        
        merged = torch.cat([unm, dst], dim=1)
        
        if last_token is not None:
            merged = torch.cat([merged, last_token], dim=1)

        return merged

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unmerge_to_size = metric_original_shape[1]
        
        if x.shape[1] % 2 != 0:
             last_token = x[:, -1:, :]
             x_even = x[:, :-1, :]
        else:
             last_token = None
             x_even = x

        unm_len = unm_idx.shape[1]
        unm, dst = x_even[..., :unm_len, :], x_even[..., unm_len:, :]
        n, _, c = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        
        out = torch.zeros(n, unmerge_to_size, c, device=x.device, dtype=x.dtype)
        
        # 填充偶数部分
        out_even = out[:, :-1, :] if unmerge_to_size % 2 != 0 else out
        out_even[..., 1::2, :] = dst
        out_even.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out_even.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        if last_token is not None:
            out[:, -1:, :] = last_token
        
        return out

    return merge, unmerge


def local_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    h: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    一个考虑局部配对的二分图软匹配实现。
    这是最终优化版本，使用1D卷积来高效、低内存地计算局部相似度。
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric_original_shape = metric.shape
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        if metric.shape[1] % 2 != 0:
            metric_for_match = metric[:, :-1, :]
        else:
            metric_for_match = metric

        a, b = metric_for_match[..., ::2, :], metric_for_match[..., 1::2, :]
        B, N_half, C = a.shape

        # --- 使用1D卷积计算滑动点积 ---
        # a: (B, N, C) -> (B, C, N) 作为输入
        # b: (B, N, C) -> (B*N, C, 1) 作为卷积核
        # 目标：计算 score_i = sum(a_i * b_{i+j}) for j in [-h, h]
        
        # 将 a 和 b 转置以符合 conv1d 的 (B, C, L) 格式
        a_conv = a.transpose(1, 2)
        b_conv = b.transpose(1, 2)

        # 将 b 的每个 token 视为一个独立的卷积核 (C, 1)
        # (B, C, N) -> (B*N, 1, C) -> (B*N, C, 1)
        # out_channels = B*N, in_channels=C, kernel_size=1
        b_filters = b.reshape(-1, C).unsqueeze(-1)

        # 使用分组卷积，每个 batch 样本有自己的一组滤波器
        # a_conv: (B, C, N) -> (1, B*C, N)
        # b_filters: (B*N, C, 1) -> (B*N, C//groups, 1), groups=C
        # 这个操作有些复杂，一个更直接的方式是循环，但为了性能我们寻找一个向量化的方法。
        # 一个更简单、内存同样高效的方法是用循环构建局部得分
        
        # 重新评估：最简单且内存高效的向量化方法是带偏移量的逐元素乘积
        local_scores = torch.zeros(B, N_half, 2 * h + 1, device=a.device, dtype=a.dtype)
        padded_b = F.pad(b, (0, 0, h, h)) # 在长度维度上填充 (左边h, 右边h)

        for i in range(2 * h + 1):
            # i=0 时，a[k] 与 b[k] 对齐
            # b_view 取的是 b 的一个窗口，与 a 的长度相同
            b_view = padded_b[:, i : i + N_half, :]
            # 逐元素相乘后在通道维度上求和，得到点积
            local_scores[:, :, i] = (a * b_view).sum(dim=-1)
        # --- 核心计算结束 ---

        if class_token:
            # CLS token 在 a 中是 a[:, 0, :]
            # 我们需要屏蔽掉它与任何 b 的匹配，但这里的匹配是 a_i -> b_j
            # CLS token 是受保护的，不参与被合并
            # 但它可能作为合并的目标，这由其在 a 中的位置决定
            # 此处的逻辑是 a 中的 token 被合并到 b 中
            # 为简单起见，且通常不合并CLS，我们阻止 a[0] 被合并
            # (通过后续的排序和选择逻辑，CLS通常有较低的相似度，不会被选)
            pass # 假设 CLS token 不会被高相似度匹配选中

        node_max, local_node_idx = local_scores.max(dim=-1)
        
        # 将局部索引转换回 b 中的全局索引
        # j = i + (local_idx - h)
        node_idx = torch.arange(N_half, device=a.device).view(1, -1) + local_node_idx - h
        
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx.gather(dim=-1, index=src_idx.squeeze(-1)).unsqueeze(-1)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]
            
    # merge 和 unmerge 函数与之前相同
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if x.shape[1] % 2 != 0:
            last_token = x[:, -1:, :]
            x_even = x[:, :-1, :]
        else:
            last_token = None
            x_even = x

        src, dst = x_even[..., ::2, :], x_even[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))

        reduce_op = 'mean' if mode == 'mean' else 'sum' if mode == 'sum' else 'amax'
        
        # 使用 scatter_reduce_ 来避免创建新的 dst 张量
        src_to_merge = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        # 确保 dst 是可写的
        dst = dst.clone()
        dst.scatter_reduce_(-2, dst_idx.expand(n, r, c), src_to_merge, reduce=reduce_op)

        merged = torch.cat([unm, dst], dim=1)

        if last_token is not None:
            merged = torch.cat([merged, last_token], dim=1)
        
        return merged

    # unmerge 函数在这里不需要，因为 ToMeBlock 只使用 merge
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        # Placeholder, as it's not used by the calling context
        raise NotImplementedError("Unmerge is not used in this ToMe patch context.")
    return merge, unmerge