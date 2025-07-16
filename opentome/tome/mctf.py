import math
import numpy as np
from typing import Callable, List, Tuple, Union
import torch.nn.functional as F
import torch
from .tome import do_nothing

"""
    MCTF specific functions
"""
def mctf_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool,
    distill_token: bool,
    attn : torch.Tensor,
    tau_sim: int=1,
    tau_info: int=1,
    tau_size: int=1,
    bidirection: int=True,
    size : torch.Tensor=None,
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
        merge: Callable, 
        x: torch.Tensor, 
        size: torch.Tensor = None,
        metric: torch.Tensor = None, 
        one_step_ahead = 0, 
        pooling_type = 'none'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    metric_n = False
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    if pooling_type == 'none':
        size_max = size.amax(dim=-2, keepdim=True)
        with torch.no_grad():
            metric_m = metric.mean(dim=[1, 2]).unsqueeze(-1)
            norm = merge(metric_m * (size / size_max), mode="sum")
        if one_step_ahead:
            metric_n = merge(metric[..., None], mode="sum").squeeze(-1)
            metric_n = merge(metric_n * metric_m[:, None] * (size / size_max)[:, None], mode="sum").squeeze(-1)
            metric_n = metric_n / norm[:, None]
        x = merge(x * metric_m * (size / size_max), mode="sum")
        with torch.no_grad():
            size = merge(size, mode="sum")
        x = x / norm
    elif pooling_type == 'max':
        x = merge(x, mode="amax")
        if one_step_ahead:
            metric_n = merge(metric[..., None], mode="sum").squeeze(-1)
            metric_n = merge(metric_n, mode="amax").squeeze(-1)
        with torch.no_grad():
            size = merge(size, mode="sum")
    elif pooling_type == 'mean':
        size_mean = torch.ones_like(size, device=x.device)
        x = merge(x, mode="sum")
        if one_step_ahead:
            metric_n = merge(metric[..., None], mode="sum").squeeze(-1)
            metric_n = merge(metric_n, mode="sum").squeeze(-1)
        with torch.no_grad():
            size = merge(size, mode="sum")
            size_mean = merge(size_mean, mode="sum")
        if one_step_ahead:
            metric_n = metric_n / size_mean[:, None]
        x = x / size_mean
    else:
        x = merge(x * size, mode="sum")
        if one_step_ahead:
            metric_n = merge(metric[..., None], mode="sum").squeeze(-1)
            metric_n = merge(metric_n * size[:, None], mode="sum").squeeze(-1)
        with torch.no_grad():
            size = merge(size, mode="sum")
        if one_step_ahead:
            metric_n = metric_n / size[:, None]
        x = x / size

    return x, size, metric_n
