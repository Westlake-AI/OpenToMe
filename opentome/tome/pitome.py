# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# PiToMe: https://github.com/hchautran/PiToMe
# --------------------------------------------------------

# ------ jinxin modified ------ #
from typing import Callable, Tuple
import torch
import torch.nn.functional as F
from opentome.tome.tome import bipartite_soft_matching


def pitome(
    metric=None,
    r:int=None,
    class_token: bool = False,
    indices:torch.Tensor=None, 
    scores:torch.Tensor=None,
) -> Tuple[Callable, Callable]:
    
    b, t, t = scores.shape
    # seperate protected token and mergeable tokens  
    merge_idx = indices[..., :2*r]
    protected_idx = indices[..., 2*r:]
    a_idx, b_idx = merge_idx[..., ::2], merge_idx[..., 1::2] 

    # get similarity scores between mergeable tokens
    scores = scores.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(b, t, r)) 
    scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(b, r, r ))
    _, dst_idx = scores.max(dim=-1) 
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if class_token:
            x_cls = x[:,0,:].unsqueeze(1)
            x = x[:,1:,:]

        b, t, c = x.shape
        batch_idx = torch.arange(b).unsqueeze_(1).to(metric.device)
        protected = x[batch_idx, protected_idx, :]
        src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
        if mode != "prune":
            dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(b, r, c), src, reduce=mode)

        if class_token:
            return torch.cat([x_cls, protected, dst], dim=1)
        return torch.cat([protected, dst], dim=1)

    return merge, None


def pitome_bipartite_soft_matching(
    metric=None,
    r:int=None,
    class_token: bool = False,
    indices:torch.Tensor=None,
    scores:torch.Tensor=None,
) -> Tuple[Callable, Callable]:

    with torch.no_grad():
        B, T, _ = scores.shape
        a_idx, b_idx = indices[..., ::2], indices[..., 1::2] 
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        scores = scores.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, b_idx.shape[-1])) 
        scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, a_idx.shape[-1], b_idx.shape[-1]))
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if class_token:
            x_cls = x[:,0,:].unsqueeze(1)
            x = x[:,1:,:]

        src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        if mode != "prune":
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if class_token:
            return torch.cat([x_cls, unm, dst], dim=1)
        return torch.cat([unm, dst], dim=1)
    
    return merge, None


def pitome_vision(
    metric: torch.Tensor, 
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    margin:torch.Tensor=0.5,
    alpha=1.0,
    use_bsm=False,
    use_bsm_pitome=False,
):
    if use_bsm:
        return bipartite_soft_matching(metric=metric, r=r, 
                        class_token=class_token, distill_token=distill_token)
    
    with torch.no_grad():
        if class_token:
            metric=metric[:,1:,:]
        # calculate energy score  
        metric = F.normalize(metric, p=2, dim=-1) 
        sim = metric@metric.transpose(-1,-2)
        energy_score = F.elu((sim - margin), alpha=alpha).mean(dim=-1)
        indices =  torch.argsort(energy_score, descending=True)
        if use_bsm_pitome:
            return pitome_bipartite_soft_matching(metric=metric, class_token=class_token, indices=indices, scores=sim, r=r)
        else:
            return pitome(metric=metric, class_token=class_token, indices=indices, scores=sim, r=r)


def pitome_text(
    metric: torch.Tensor, 
    r: int,
    margin:torch.Tensor=0.5,
    class_token: bool = False,
):
    with torch.no_grad():
        if class_token:
            metric=metric[:,1:,:]

        if len(metric.shape) == 2:
            metric = metric[None,...]
        b, t, c = metric.shape
        metric = F.normalize(metric, p=2, dim=-1) 
        batch_idx = torch.arange(b).unsqueeze_(1).to(metric.device)
        # To calculate energy scores for text tokens, in this implementation, we use the Gaussian kernel.
        # This shows better performance than the equation (4) in the paper 
        sim = metric@metric.transpose(-1,-2)
        # sim = F.elu((metric@metric.transpose(-1,-2) - margin)/0.01, alpha=alpha)
        sigma = 1 - margin 
        energy_score = (torch.exp(-(((1 - sim)/sigma)**2 * 0.5))).mean(-1) *  1/(sigma*torch.sqrt(torch.tensor(2*torch.pi))) 
        indices =  torch.argsort(energy_score , descending=True)
        merge_idx = indices[..., :2 * r]
        protected_idx = indices[..., 2 * r:]
        # Also instead of using odd and even indices,
        # we choose to split based on higher and lower energy sets which show significantly better performance 
        a_idx, b_idx = merge_idx[..., :r], merge_idx[..., r:]
        scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(b, t, r)) 
        scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(b, r, r ))
        _, dst_idx = scores.max(dim=-1) 

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if class_token:
            x_cls = x[:,0,:].unsqueeze(1)
            x = x[:,1:,:]
        else:
            x_cls = None

        b, _, c = x.shape
        protected = x[batch_idx, protected_idx, :]
        src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
        dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(b, r, c), src, reduce=mode)

        if class_token:
            return torch.cat([x_cls, protected, dst], dim=1)
        return torch.cat([protected, dst], dim=1)

    return merge


def merge_mean(
    merge: Callable, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    x = merge(x, mode="mean")
    return x


def prune(
    merge: Callable, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    x = merge(x, mode="prune")
    return x
