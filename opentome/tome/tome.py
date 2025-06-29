# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
import numpy as np
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
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

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


"""
    MCTF specific functions
"""
def mctf_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool,
    distill_token: bool,
    tau_sim: int=1,
    tau_info: int=1,
    tau_size: int=1,
    bidirection: int=True,
    size : torch.Tensor=None,
    attn : torch.Tensor=None,
) -> Tuple[Callable, Callable]:

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

    if bidirection:
        r1, r2 = r // 2, r - r // 2
    else:
        r1, r2 = r, 0

    B, T, _ = metric.shape
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        if tau_sim:
            W_sim = a @ b.transpose(-1, -2)
            W_sim = ((W_sim + 1) / 2) ** (1 / tau_sim)
        else:
            W_sim = torch.ones((a.shape[0], a.shape[1], b.shape[1]), device=a.device)

        if tau_info > 0 and attn is not None:
            attn = 1 / attn.mean(dim=[1, 2])
            attn = attn / attn.max(1, keepdim=True)[0]
            attn_a, attn_b = attn[..., ::2, None], attn[..., 1::2, None].transpose(1, 2)
            W_info = (attn_a * attn_b) ** (1 / tau_info)
        else:
            W_info = 1

        if tau_size and size is not None:
            size = 1 / size
            size = size / size.max(1, keepdim=True)[0]
            size_a, size_b = size[..., ::2, :], size[..., 1::2, :].transpose(1, 2)

            W_size = (size_a * size_b) ** (1 / tau_size)
        else:
            W_size = 1

        scores = W_sim * W_info * W_size

        if class_token:
            scores[..., 0, :] = -math.inf

        n, t1, t2 = scores.shape

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r1:, :]
        src_idx = edge_idx[..., :r1, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

        #### bidirection:
        new_scores = scores.gather(dim=-2, index=unm_idx.repeat(1, 1, t2)).transpose(1, 2)

        node_max2, node_idx2 = new_scores.max(dim=-1)
        edge_max2, edge_idx2 = node_max2.sort(dim=-1, descending=True)
        edge_idx2 = edge_idx2[..., None]
        unm_idx2 = edge_idx2[..., r2:, :]
        src_idx2 = edge_idx2[..., :r2, :]
        dst_idx2 = node_idx2[..., None].gather(dim=-2, index=src_idx2)

    def dim_match(src, dim=1, dim_num=5):
        while len(src.shape) < dim_num:
            src = src.unsqueeze(dim)
        return src

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]

        b, mid, c = src.shape[0], src.shape[1:-2], src.shape[-1]
        dim_num = len(src.shape)
        unm_idx_ = dim_match(unm_idx, dim=1, dim_num=dim_num)
        src_idx_ = dim_match(src_idx, dim=1, dim_num=dim_num)
        dst_idx_ = dim_match(dst_idx, dim=1, dim_num=dim_num)
        unm = src.gather(dim=-2, index=unm_idx_.expand(b, *mid, t1 - r1, c))
        src = src.gather(dim=-2, index=src_idx_.expand(b, *mid, r1, c))
        dst = dst.scatter_reduce(-2, dst_idx_.expand(b, *mid, r1, c), src, reduce=mode)
        if bidirection:
            unm_idx2_ = dim_match(unm_idx2, dim=1, dim_num=dim_num)
            src_idx2_ = dim_match(src_idx2, dim=1, dim_num=dim_num)
            dst_idx2_ = dim_match(dst_idx2, dim=1, dim_num=dim_num)

            src2, dst2 = dst, unm
            unm2 = src2.gather(dim=-2, index=unm_idx2_.expand(b, *mid, t2 - r2, c))
            src2 = src2.gather(dim=-2, index=src_idx2_.expand(b, *mid, r2, c))
            dst2 = dst2.scatter_reduce(-2, dst_idx2_.expand(b, *mid, r2, c), src2, reduce=mode)
            x = torch.cat([dst2, unm2], dim=-2)
        else:
            x = torch.cat([unm, dst], dim=-2)

        return x
    
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r1, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r1, c), src=src)

        return out

    return merge, unmerge


def mctf_merge_wavg(
        merge: Callable, x: torch.Tensor, attn: torch.Tensor, size: torch.Tensor = None, 
        one_step_ahead = 0, pooling_type = 'none'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    attn_n = False
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    if pooling_type == 'none':
        size_max = size.amax(dim=-2, keepdim=True)
        with torch.no_grad():
            attn_m = attn.mean(dim=[1, 2]).unsqueeze(-1)
            norm = merge(attn_m * (size / size_max), mode="sum")
        if one_step_ahead:
            attn_n = merge(attn[..., None], mode="sum").squeeze(-1)
            attn_n = merge(attn_n * attn_m[:, None] * (size / size_max)[:, None], mode="sum").squeeze(-1)
            attn_n = attn_n / norm[:, None]
        x = merge(x * attn_m * (size / size_max), mode="sum")
        with torch.no_grad():
            size = merge(size, mode="sum")
        x = x / norm
    elif pooling_type == 'max':
        x = merge(x, mode="amax")
        if one_step_ahead:
            attn_n = merge(attn[..., None], mode="sum").squeeze(-1)
            attn_n = merge(attn_n, mode="amax").squeeze(-1)
        with torch.no_grad():
            size = merge(size, mode="sum")
    elif pooling_type == 'mean':
        size_mean = torch.ones_like(size, device=x.device)
        x = merge(x, mode="sum")
        if one_step_ahead:
            attn_n = merge(attn[..., None], mode="sum").squeeze(-1)
            attn_n = merge(attn_n, mode="sum").squeeze(-1)
        with torch.no_grad():
            size = merge(size, mode="sum")
            size_mean = merge(size_mean, mode="sum")
        if one_step_ahead:
            attn_n = attn_n / size_mean[:, None]
        x = x / size_mean
    else:
        x = merge(x * size, mode="sum")
        if one_step_ahead:
            attn_n = merge(attn[..., None], mode="sum").squeeze(-1)
            attn_n = merge(attn_n * size[:, None], mode="sum").squeeze(-1)
        with torch.no_grad():
            size = merge(size, mode="sum")
        if one_step_ahead:
            attn_n = attn_n / size[:, None]
        x = x / size

    return x, size, attn_n


"""
    DCT specific functions
"""
def dc_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
) -> Tuple[Callable, Callable]:
    
    def dct(metric, norm=None):

        metric_shape = metric.shape
        n = metric_shape[-1]
        metric = metric.contiguous().view(-1, n)

        v = torch.cat([metric[:, ::2], metric[:, 1::2].flip([1])], dim=1)
        Vc = torch.fft.fft(v, dim=1)

        k = - torch.arange(n, dtype=metric.dtype, device=metric.device)[None, :] * np.pi / (2 * n)
        V = Vc.real * torch.cos(k) - Vc.imag * torch.sin(k)

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(n) * 2
            V[:, 1:] /= np.sqrt(n / 2) * 2
        V = 2 * V.view(*metric_shape)
        return V
    
    def idct(metric, norm=None):

        metric_shape = metric.shape
        n = metric_shape[-1]
        metric = metric.contiguous().view(-1, n) / 2

        if norm == 'ortho':
            metric[:, 0] *= np.sqrt(n) * 2
            metric[:, 1:] *= np.sqrt(n / 2) * 2

        k = torch.arange(n, dtype = metric.dtype, device = metric.device)[None, :] * np.pi / (2 * n)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = metric
        V_t_i = torch.cat([metric[:, :1] * 0, -metric.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
        V = torch.view_as_complex(V)

        v = torch.fft.ifft(V, dim=1).real
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :n - (n // 2)]
        x[:, 1::2] += v.flip([1])[:, :n // 2]

        return x.view(*metric_shape)
    
    if class_token:
        cls = metric[:, 0, :].unsqueeze_(1)
        metric = metric[:, 1:, :]
    t = metric.shape[1]
    r = min(r, t // 2)

    metric = metric.type(torch.float32).permute(1, 0, 2)

    dct_metric = dct(metric.transpose(0, 2), norm='ortho').transpose(0, 2)

    # feel free to play with any method here
    if r is not None: 
        dct_metric = dct_metric[:t - r, :, :]

    metric = idct(dct_metric.transpose(0, 2), norm='ortho').transpose(0, 2).permute(1, 0, 2)

    if class_token:
        return torch.cat([cls, metric], dim=1)
    return metric


"""
    CrossGet still in development, not used in the paper yet
""" 
def crossget_bipartite_soft_matching(
    r: int,
    metric: torch.Tensor,
    query_token: torch.Tensor,
    class_token: bool = False,
    distill_token: bool = False,
):
    
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    b, t, _ = metric.shape
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing
    
    with torch.no_grad():

        metric = metric / metric.norm(dim=-1, keepdim=True)
        score = metric @ metric.transpose(-1, -2)
        score += torch.empty_like(score).fill_(-math.inf).tril_().triu_()
        if class_token:
            score[:, 0, :] = score[:, :, 0] = -math.inf
            score[:, -1, :] = score[:, :, -1] = -math.inf
        else:
            score[:, 0, :] = score[:, :, 0] = -math.inf

        edge_idx = torch.max(score, dim=-1, keepdim=True)[0].argsort(dim=1, descending=True).expand(-1, -1, t)
        mask = score.scatter(1, edge_idx, torch.empty_like(score).fill_(-math.inf).tril_())
        mask = score.scatter(-1, edge_idx.transpose(-1, -2), mask)
        importance = torch.mean(query_token[..., :-1], dim=-1, keepdim=True) + query_token[..., -1, None] 
        edge_idx = (importance - torch.max(score + mask, dim=-1, keepdim=True)[0]).argsort(dim=1, descending=False)
        src_idx, dst_all_idx = edge_idx[..., :r, :], edge_idx[..., r:, :]  
        
        score = score.gather(dim=1, index=src_idx.expand(-1, -1, t)).gather(dim=-1, index=dst_all_idx.transpose(-1, -2).expand(-1, r, -1))
        dst_idx = score.argmax(dim=-1)[..., None]
        weight_src = importance.gather(dim=1, index=src_idx)
        weight_dst = importance.gather(dim=1, index=dst_all_idx).gather(dim=1, index=dst_idx)
        weight = F.softmax(torch.cat([weight_src, weight_dst], dim=-1), dim=-1)

        def crossget_merge(x):
            c = x.shape[-1]
            src = x.gather(dim=1, index=src_idx.expand(-1, -1, c))
            dst = x.gather(dim=1, index=dst_all_idx.expand(-1, -1, c))
            if weight is not None:
                weight *= weight.shape[-1]
                dst = dst.scatter_add(1, dst_idx.expand(-1, -1, c), weight[..., 0, None] * src + \
                     (weight[..., 1, None] - 1) * dst.gather(dim=1, index=dst_idx.expand(-1, -1, c)))
            else:
                dst = dst.scatter_add(1, dst_idx.expand(-1, -1, c), src)
            out = dst.gather(dim=1, index=dst_all_idx.argsort(dim=1).expand(-1, -1, c)) # gather for keeping order
            return out

        return crossget_merge, weight


def cross_merge_wavg(
    crossget_merge: Callable, x: torch.Tensor, size: torch.Tensor = None, weight: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = crossget_merge(x * size, weight)
    size = crossget_merge(size)

    x = x / size
    return x, size
