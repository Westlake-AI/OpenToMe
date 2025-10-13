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


def map_to_one_hot(token_map, t_new, device):
    # Converts a token_map to a one-hot source matrix for verification.
    b, t_orig = token_map.shape
    one_hot = torch.zeros(b, t_new, t_orig, device=device)
    b_idx = torch.arange(b, device=device)[:, None].expand(-1, t_orig)
    t_orig_idx = torch.arange(t_orig, device=device)[None, :].expand(b, -1)
    new_indices = token_map
    one_hot[b_idx, new_indices, t_orig_idx] = 1.0
    return one_hot


def merge_source_matrix(
    merge: Callable,
    x: torch.Tensor,
    source: torch.Tensor = None
) -> torch.Tensor:
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
    source = merge(source, mode="amax")

    return source


def merge_source_map(
    current_level_map: torch.Tensor,
    x: torch.Tensor,
    source_map: torch.Tensor = None
) -> torch.Tensor:
    if source_map is None:
        return current_level_map
    updated_source_map = torch.gather(current_level_map, 1, source_map)

    return updated_source_map


"""
    ToMe/ToFu/DETM specific functions
"""
def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable, torch.Tensor]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).
    This version includes a robust `merge` function compatible with the original
    `merge_source` logic for 'matrix' mode.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    n, t = metric.shape[0], metric.shape[1]
    r = min(r, (t - protected) // 2)

    # A simple do_nothing function for the r=0 case
    do_nothing = lambda x, **_: x

    if r <= 0:
        identity_map = torch.arange(t, device=metric.device, dtype=torch.long).expand(n, -1)
        return do_nothing, do_nothing, identity_map

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
            unm_idx = unm_idx.sort(dim=1)[0]
        
        # This part for building token_map is correct and remains unchanged
        t_orig = t
        t_new = t_orig - r
        token_map = torch.empty((n, t_orig), device=metric.device, dtype=torch.long)
        
        unm_tokens_orig = (2 * unm_idx.squeeze(-1)).long()
        num_unm = unm_tokens_orig.shape[-1]
        unm_new_indices = torch.arange(num_unm, device=metric.device).expand(n, -1)
        token_map.scatter_(1, unm_tokens_orig, unm_new_indices)
        
        dst_all_orig = torch.arange(1, t_orig, 2, device=metric.device, dtype=torch.long).expand(n, -1)
        dst_all_new_indices = torch.arange(num_unm, t_new, device=metric.device).expand(n, -1)
        token_map.scatter_(1, dst_all_orig, dst_all_new_indices)
        
        src_tokens_orig = (2 * src_idx.squeeze(-1)).long()
        dst_tokens_target_orig = (2 * dst_idx.squeeze(-1) + 1).long()
        target_new_indices = torch.gather(token_map, 1, dst_tokens_target_orig)
        token_map.scatter_(1, src_tokens_orig, target_new_indices)
    
    # --- ROBUST MERGE FUNCTION ---
    # This is the key correction.
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1-r, c))
        if mode == 'amax':
            src_gathered = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src_gathered, reduce='amax')
        elif mode == 'mean':
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
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        # (This function remains unchanged, but corrected for clarity.)
        unm_len = unm_idx.shape[1]
        unm, dst_merged = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        # Create a new 'b' set, filling in the merged tokens
        b = torch.zeros(n, metric.shape[1] // 2, c, device=x.device, dtype=x.dtype)
        b.scatter_(dim=-2, index=dst_idx.expand(n, r, c), src=dst_merged)
        
        # Gather the source tokens that were merged
        src_merged = b.gather(dim=-2, index=dst_idx.expand(n, r, c))

        # Reconstruct the original tensor
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        # Place back the 'b' set
        out[..., 1::2, :] = b
        # Scatter back the unmerged and merged 'a' tokens
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src_merged)

        return out

    return merge, unmerge, token_map


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


def naive_local_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    h: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable, torch.Tensor]:
    """
    A naive implementation of bipartite soft matching with locality.
    This version computes the full score matrix 1and then applies a mask.
    It now correctly generates and returns a `token_map` for source tracking.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    n, t = metric.shape[0], metric.shape[1]
    r = min(r, (t - protected) // 2)

    # If r is 0, do nothing and return an identity map
    if r <= 0:
        identity_map = torch.arange(t, device=metric.device, dtype=torch.long).expand(n, -1)
        return do_nothing, do_nothing, identity_map

    with torch.no_grad():
        metric_original_shape = metric.shape
        metric = metric / metric.norm(dim=-1, keepdim=True)

        # Handle odd-length sequences by excluding the last token from matching
        if metric.shape[1] % 2 != 0:
            metric_for_match = metric[:, :-1, :]
        else:
            metric_for_match = metric
        
        a, b = metric_for_match[..., ::2, :], metric_for_match[..., 1::2, :]
        
        scores = a @ b.transpose(-1, -2)
        
        # Apply locality mask
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
        
        # --- TOKEN_MAP GENERATION ---
        t_even = metric_for_match.shape[1]
        t_new_even = t_even - r
        
        token_map_even = torch.empty((n, t_even), device=metric.device, dtype=torch.long)

        unm_tokens_orig = (2 * unm_idx.squeeze(-1)).long()
        num_unm = unm_tokens_orig.shape[-1]
        unm_new_indices = torch.arange(num_unm, device=metric.device).expand(n, -1)
        token_map_even.scatter_(1, unm_tokens_orig, unm_new_indices)
        
        dst_all_orig = torch.arange(1, t_even, 2, device=metric.device, dtype=torch.long).expand(n, -1)
        dst_all_new_indices = torch.arange(num_unm, t_new_even, device=metric.device).expand(n, -1)
        token_map_even.scatter_(1, dst_all_orig, dst_all_new_indices)
        
        src_tokens_orig = (2 * src_idx.squeeze(-1)).long()
        dst_tokens_target_orig = (2 * dst_idx.squeeze(-1) + 1).long()
        target_new_indices = torch.gather(token_map_even, 1, dst_tokens_target_orig)
        token_map_even.scatter_(1, src_tokens_orig, target_new_indices)

        if t % 2 != 0:
            token_map = torch.empty((n, t), device=metric.device, dtype=torch.long)
            token_map[:, :-1] = token_map_even
            token_map[:, -1] = t_new_even
        else:
            token_map = token_map_even
        # --- END OF TOKEN_MAP GENERATION ---

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

        src_gathered = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src_gathered, reduce=mode)
        
        merged = torch.cat([unm, dst], dim=1)
        
        if last_token is not None:
            merged = torch.cat([merged, last_token], dim=1)

        return merged

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unmerge_to_size = metric_original_shape[1]
        
        has_last_token = unmerge_to_size % 2 != 0
        if has_last_token:
            last_token = x[:, -1:, :]
            x_even = x[:, :-1, :]
        else:
            last_token = None
            x_even = x

        unm_len = unm_idx.shape[1]
        unm, dst_merged = x_even[..., :unm_len, :], x_even[..., unm_len:, :]
        n, _, c = unm.shape

        b = dst_merged
        src_merged = b.gather(dim=-2, index=dst_idx.expand(n, r, c))
        
        a = torch.zeros(n, unm_len + r, c, device=x.device, dtype=x.dtype)
        a.scatter_(dim=-2, index=unm_idx.expand(n, unm_len, c), src=unm)
        a.scatter_(dim=-2, index=src_idx.expand(n, r, c), src=src_merged)

        out = torch.zeros(n, unmerge_to_size, c, device=x.device, dtype=x.dtype)
        
        # Handle the even part of the sequence first
        if has_last_token:
            out_even = out[:, :-1, :]
        else:
            out_even = out
            
        out_even[..., ::2, :] = a
        out_even[..., 1::2, :] = b

        if has_last_token:
            out[:, -1:, :] = last_token
        
        return out

    return merge, unmerge, token_map


def local_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    h: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable, torch.Tensor]:
    """
    An efficient implementation of bipartite soft matching with locality using 1D convolution.
    It now correctly generates and returns a `token_map` for source tracking.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    n, t = metric.shape[0], metric.shape[1]
    r = min(r, (t - protected) // 2, h-5)

    if r <= 0:
        identity_map = torch.arange(t, device=metric.device, dtype=torch.long).expand(n, -1)
        return do_nothing, do_nothing, identity_map

    with torch.no_grad():
        metric_original_shape = metric.shape
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        if metric.shape[1] % 2 != 0:
            metric_for_match = metric[:, :-1, :]
        else:
            metric_for_match = metric

        a, b = metric_for_match[..., ::2, :], metric_for_match[..., 1::2, :]
        B, N_half, C = a.shape
        
        local_scores = torch.zeros(B, N_half, 2 * h + 1, device=a.device, dtype=a.dtype)
        padded_b = F.pad(b.transpose(-1, -2), (h, h), mode='replicate').transpose(-1, -2)

        for i in range(2 * h + 1):
            b_view = padded_b[:, i : i + N_half, :]
            local_scores[:, :, i] = (a * b_view).sum(dim=-1)

        if class_token:
            local_scores[:, 0, :] = -math.inf

        node_max, local_node_idx = local_scores.max(dim=-1)
        
        # --- MODIFICATION: Clamp indices to prevent out-of-bounds error ---
        # Convert local indices to global indices
        node_idx = torch.arange(N_half, device=a.device).view(1, -1) + local_node_idx - h
        # Clamp the indices to be within the valid range [0, N_half - 1]
        node_idx.clamp_(0, N_half - 1)
        # --- END OF MODIFICATION ---
        
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx.gather(dim=-1, index=src_idx.squeeze(-1)).unsqueeze(-1)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]
        
        # --- TOKEN_MAP GENERATION ---
        t_even = metric_for_match.shape[1]
        t_new_even = t_even - r
        
        token_map_even = torch.empty((n, t_even), device=metric.device, dtype=torch.long)

        unm_tokens_orig = (2 * unm_idx.squeeze(-1)).long()
        num_unm = unm_tokens_orig.shape[-1]
        unm_new_indices = torch.arange(num_unm, device=metric.device).expand(n, -1)
        token_map_even.scatter_(1, unm_tokens_orig, unm_new_indices)
        
        dst_all_orig = torch.arange(1, t_even, 2, device=metric.device, dtype=torch.long).expand(n, -1)
        dst_all_new_indices = torch.arange(num_unm, t_new_even, device=metric.device).expand(n, -1)
        token_map_even.scatter_(1, dst_all_orig, dst_all_new_indices)
        
        src_tokens_orig = (2 * src_idx.squeeze(-1)).long()
        dst_tokens_target_orig = (2 * dst_idx.squeeze(-1) + 1).long()
        target_new_indices = torch.gather(token_map_even, 1, dst_tokens_target_orig)
        token_map_even.scatter_(1, src_tokens_orig, target_new_indices)

        if t % 2 != 0:
            token_map = torch.empty((n, t), device=metric.device, dtype=torch.long)
            token_map[:, :-1] = token_map_even
            token_map[:, -1] = t_new_even
        else:
            token_map = token_map_even
        # --- END OF TOKEN_MAP GENERATION ---

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

        src_gathered = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src_gathered, reduce=mode)
        
        merged = torch.cat([unm, dst], dim=1)
        
        if last_token is not None:
            merged = torch.cat([merged, last_token], dim=1)

        return merged

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unmerge_to_size = metric_original_shape[1]
        
        has_last_token = unmerge_to_size % 2 != 0
        if has_last_token:
            last_token = x[:, -1:, :]
            x_even = x[:, :-1, :]
        else:
            last_token = None
            x_even = x

        unm_len = unm_idx.shape[1]
        unm, dst_merged = x_even[..., :unm_len, :], x_even[..., unm_len:, :]
        n, _, c = unm.shape

        b = dst_merged
        src_merged = b.gather(dim=-2, index=dst_idx.expand(n, r, c))
        
        a = torch.zeros(n, unm_len + r, c, device=x.device, dtype=x.dtype)
        a.scatter_(dim=-2, index=unm_idx.expand(n, unm_len, c), src=unm)
        a.scatter_(dim=-2, index=src_idx.expand(n, r, c), src=src_merged)

        out = torch.zeros(n, unmerge_to_size, c, device=x.device, dtype=x.dtype)

        if has_last_token:
            out_even = out[:, :-1, :]
        else:
            out_even = out
            
        out_even[..., ::2, :] = a
        out_even[..., 1::2, :] = b

        if has_last_token:
            out[:, -1:, :] = last_token
        
        return out

    return merge, unmerge, token_map


def token_unmerge(merged_tokens, source=None):
    if source is None:
        return merged_tokens
    B, _, C = merged_tokens.shape
    full_L = source.shape[-1]  # [B, keep_L, full_L]
    full_tokens = torch.zeros(B, full_L, C, device=merged_tokens.device, dtype=merged_tokens.dtype)
    indices = source.nonzero(as_tuple=False)
    batch_idx = indices[:, 0]
    full_tokens[batch_idx, indices[:, 2], :] = merged_tokens[batch_idx, indices[:, 1], :]
    return full_tokens


def token_unmerge_from_map(merged_tokens, token_map=None):
    if token_map is None:
        return merged_tokens
    B= merged_tokens.shape[0]
    T_full = token_map.shape[1]
    b_idx = torch.arange(B, device=merged_tokens.device, dtype=torch.int)[:, None].expand(-1, T_full)
    full_tokens = merged_tokens[b_idx, token_map]
    return full_tokens