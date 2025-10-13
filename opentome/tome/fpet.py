# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# FPET: https://github.com/kyk120/fpet
# --------------------------------------------------------

# ------ jinxin modified ------ #
import math
from typing import Callable, Tuple

import torch
import math


def do_nothing(x, mode=None):
    return x


def checkerboard_split(x: torch.Tensor, protected: int):
    
    x_others = x
    
    if protected:
        x_cls = x_others[:, :1, :]
        x_others = x_others[:, 1:, :]
    if protected == 2:
        x_distill = x_others[:, -1:, :]
        x_others = x_others[:, :-1, :]

    B, N, C = x_others.shape

    if N == 196:
        width = 14
        height = int(N/width)
    else:
        width = int(math.floor(math.sqrt(N)))
        height = int(N/width)

    x_others = x_others.view(B, height, width, C)

    x_tl = x_others[..., ::2, :][:, ::2, :, :].reshape(B, -1, C)
    x_br = x_others[..., 1::2, :][:, 1::2, :, :].reshape(B, -1, C)

    x_tr = x_others[..., ::2, :][:, 1::2, :, :].reshape(B, -1, C)
    x_bl = x_others[..., 1::2, :][:, ::2, :, :].reshape(B, -1, C)

    a = torch.cat([x_tl, x_br], dim=1)
    b_tensors = [x_tr, x_bl]
    if protected:
        b_tensors = [x_cls] + b_tensors
    if protected == 2:
        b_tensors = b_tensors + [x_distill]
    b = torch.cat(b_tensors, dim=1)

    return a, b


def generate_checkerboard_token_map(x: torch.Tensor, protected: int) -> torch.Tensor:
    """
    Generate token map for checkerboard pattern merging.
    
    Args:
        x: Input tensor of shape (batch, tokens, channels)
        protected: Number of protected tokens (class and distill tokens)
        
    Returns:
        token_map: Mapping from original token indices to new token indices
    """
    B, N, C = x.shape
    
    # Initialize token map with identity mapping
    token_map = torch.arange(N, device=x.device, dtype=torch.long).expand(B, -1)
    
    if protected:
        x_others = x[:, 1:, :] if protected >= 1 else x
        if protected == 2:
            x_others = x_others[:, :-1, :]
    else:
        x_others = x
    
    N_others = x_others.shape[1]
    
    if N_others == 196:
        width = 14
        height = int(N_others/width)
    else:
        width = int(math.floor(math.sqrt(N_others)))
        height = int(N_others/width)
    
    # Create grid indices for the checkerboard pattern
    h_indices, w_indices = torch.meshgrid(
        torch.arange(height, device=x.device), 
        torch.arange(width, device=x.device), 
        indexing='ij'
    )
    
    # Flatten indices
    h_flat = h_indices.reshape(-1)
    w_flat = w_indices.reshape(-1)
    
    # Determine which tokens belong to which groups
    is_tl_br = ((h_flat % 2) == (w_flat % 2))  # Top-left and bottom-right
    is_tr_bl = ((h_flat % 2) != (w_flat % 2))  # Top-right and bottom-left
    
    # Create mapping for checkerboard pattern
    # For simplicity, we'll map tokens in the tr_bl group to the tl_br group
    tl_br_indices = torch.where(is_tl_br)[0]
    tr_bl_indices = torch.where(is_tr_bl)[0]
    
    # In checkerboard merging, we merge tr_bl tokens into tl_br tokens
    # So we need to map tr_bl indices to their corresponding tl_br indices
    if len(tr_bl_indices) > 0 and len(tl_br_indices) > 0:
        # Create a mapping from tr_bl positions to tl_br positions
        min_len = min(len(tr_bl_indices), len(tl_br_indices))
        
        # For each tr_bl token, map it to the corresponding tl_br token
        for i in range(min_len):
            tr_bl_pos = tr_bl_indices[i] + protected  # Adjust for protected tokens
            tl_br_pos = tl_br_indices[i] + protected  # Adjust for protected tokens
            if tr_bl_pos < N and tl_br_pos < N:
                token_map[:, tr_bl_pos] = tl_br_pos
    
    return token_map


def fpet_bipartite_diff_matching(
    metric: torch.Tensor,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, torch.Tensor]:
    """
    Input size is [batch, tokens, channels].

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    
    Returns:
     - merge: A callable merge function.
     - token_map: A tensor mapping old token indices to new ones for source tracking.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # with torch.no_grad():
    metric = metric / metric.norm(dim=-1, keepdim=True)
    a, b = checkerboard_split(metric, protected)

    scores = a @ b.transpose(-1, -2)

    if class_token:
        scores[..., :, 0] = -math.inf
    if distill_token:
        scores[..., :, 0] = -math.inf

    v, idx = torch.topk(scores, 2, dim=-1)
    mean12 = v.mean(dim=-1, keepdim=True)
    soft_matrix = torch.sigmoid(scores - mean12)
    hard_matrix = (soft_matrix > 0.5).float()
    matching_matrix = soft_matrix + (hard_matrix - soft_matrix).detach()

    # Generate token map for source tracking based on checkerboard pattern
    token_map = generate_checkerboard_token_map(metric, protected)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        x_a, x_b = checkerboard_split(x, protected)

        if mode == "sum":
            x_merge = torch.einsum('bik, bij->bkj', matching_matrix, x_a)
            x_merged_sum = x_b + x_merge
        elif mode == "amax":
            x_merge = torch.einsum('bik, bij->bkj', matching_matrix, x_a)
            x_merged_sum = torch.max(x_b, x_merge)
        else:
            x_merge = torch.einsum('bik, bij->bkj', matching_matrix, x_a)
            x_merged_sum = x_b + x_merge
            
        return x_merged_sum
    
    return merge, token_map