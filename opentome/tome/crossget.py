import math
import numpy as np
from typing import Callable, List, Tuple, Union
import torch.nn.functional as F
import torch
from .tome import do_nothing

"""
    CrossGet still in development, not used in the paper yet
""" 
def crossget_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool,
    distill_token: bool,
    query_token: torch.Tensor,
)-> Tuple[Callable, Callable]:
    
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
    crossget_merge: Callable, 
    x: torch.Tensor, 
    size: torch.Tensor = None, 
    weight: torch.Tensor = None
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
