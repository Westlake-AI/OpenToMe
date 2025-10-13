# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MCTF: https://github.com/mlvlab/MCTF
# --------------------------------------------------------

# ------ jinxin modified ------ #
import math
from typing import Callable, List, Tuple, Union
import torch
from .tome import do_nothing



def _build_token_map_bidirectional(
    n: int,
    t_orig: int,
    t1: int,
    t2: int,
    r1: int,
    r2: int,
    unm_idx: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    bidirection: bool,
    device: torch.device,
    dtype: torch.dtype,
    unm_idx2: torch.Tensor = None,
    src_idx2: torch.Tensor = None,
    dst_idx2: torch.Tensor = None,
) -> torch.Tensor:

    num_unm = t1 - r1
    t_new_inter = num_unm + t2
    token_map_inter = torch.empty((n, t_orig), device=device, dtype=torch.long)

    # a Merged tokens map to [0 .. num_unm-1]
    unm_tokens_orig = (2 * unm_idx.squeeze(-1)).long()
    unm_new_indices = torch.arange(num_unm, device=device).expand(n, -1)
    token_map_inter.scatter_(1, unm_tokens_orig, unm_new_indices)

    # All b (odd positions) map to [num_unm .. num_unm + t2 - 1]
    dst_all_orig = torch.arange(1, t_orig, 2, device=device, dtype=torch.long).expand(n, -1)
    dst_all_new_indices = torch.arange(num_unm, num_unm + t2, device=device).expand(n, -1)
    token_map_inter.scatter_(1, dst_all_orig, dst_all_new_indices)

    # a Merged tokens map to the middle index of their target b
    src_tokens_orig = (2 * src_idx.squeeze(-1)).long()
    dst_tokens_target_orig = (2 * dst_idx.squeeze(-1) + 1).long()
    target_new_indices = torch.gather(token_map_inter, 1, dst_tokens_target_orig)
    token_map_inter.scatter_(1, src_tokens_orig, target_new_indices)

    # Intermediate to final
    inter_to_final = torch.arange(t_new_inter, device=device, dtype=torch.long).expand(n, -1).clone()
    if bidirection and r2 > 0:
        # b Unmerged positions (sorted by unm_idx2)
        pos_in_unm2 = torch.full((n, t2), -1, device=device, dtype=torch.long)
        order_unm2 = torch.arange(t2 - r2, device=device).expand(n, -1)
        pos_in_unm2.scatter_(1, unm_idx2.squeeze(-1), order_unm2)

        # b Final indices: compact b in [num_unm .. num_unm + (t2 - r2) - 1]
        final_idx_b = torch.where(
            pos_in_unm2 >= 0,
            num_unm + pos_in_unm2,
            torch.zeros(1, device=device, dtype=torch.long).expand(n, t2),
        )

        # Merged b (src_idx2) map to their target a Unmerged positions (dst_idx2)
        final_idx_b.scatter_(1, src_idx2.squeeze(-1), dst_idx2.squeeze(-1))

        # Write back to the b segment of inter_to_final
        inter_to_final[:, num_unm:num_unm + t2] = final_idx_b

    # Combine: original -> intermediate -> final
    token_map = inter_to_final.gather(1, token_map_inter)
    return token_map


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
) -> Tuple[Callable, Callable, torch.Tensor]:

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    n, t = metric.shape[0], metric.shape[1]
    r = min(r, (t - protected) // 2)

    # A simple do_nothing function for the r=0 case
    do_nothing = lambda x, **_: x

    if r <= 0:
        identity_map = torch.arange(t, device=metric.device, dtype=torch.long).expand(n, -1)
        return do_nothing, do_nothing, identity_map

    if bidirection:
        r1, r2 = r // 2, r - r // 2
    else:
        r1, r2 = r, 0

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

        #####################
        #### Bidirection ####
        #####################
        new_scores = scores.gather(dim=-2, index=unm_idx.repeat(1, 1, t2)).transpose(1, 2)

        node_max2, node_idx2 = new_scores.max(dim=-1)
        edge_max2, edge_idx2 = node_max2.sort(dim=-1, descending=True)
        edge_idx2 = edge_idx2[..., None]
        unm_idx2 = edge_idx2[..., r2:, :]
        src_idx2 = edge_idx2[..., :r2, :]
        dst_idx2 = node_idx2[..., None].gather(dim=-2, index=src_idx2)

        # Support token merged by map, instead of matrix
        t_orig = t
        t1 = a.shape[1]
        t2 = b.shape[1]
        token_map = _build_token_map_bidirectional(
            n=n,
            t_orig=t_orig,
            t1=t1,
            t2=t2,
            r1=r1,
            r2=r2,
            unm_idx=unm_idx,
            src_idx=src_idx,
            dst_idx=dst_idx,
            bidirection=bidirection,
            device=metric.device,
            dtype=metric.dtype,
            unm_idx2=unm_idx2,
            src_idx2=src_idx2,
            dst_idx2=dst_idx2,
        )

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

    return merge, unmerge, token_map


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
