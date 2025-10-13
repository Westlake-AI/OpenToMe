# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

# ------ jinxin modified ------ #
from typing import Tuple

import torch
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock

from opentome.tome.fpet import fpet_bipartite_diff_matching
from opentome.tome.tome import (
    merge_source_matrix,
    merge_source_map,
    merge_wavg,
)
# from adaptformer import Adapter
from opentome.timm import Attention, Block


class FPETBlock(Block):
    """
    Modifications:
     - Apply FPET between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        assert isinstance(metric['metric'], (float, torch.Tensor)), "metric not a float or torch.Tensor"
        x = x + self._drop_path1(x_attn)
        if hasattr(self, 'refinement'):
            # TODO
            metric['metric'] = metric['metric'].detach()
            metric['metric'] = metric['metric'] + 1 * self.refinement(metric['metric'])
            raise ValueError("[Warning] We do not support this module for inference.")
        else:
            merge, current_level_map = fpet_bipartite_diff_matching(
                metric['metric'], self._tome_info["class_token"], self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                if self._tome_info["source_tracking_mode"] == 'map':
                    source_map = self._tome_info["source_map"]
                    # Initialize map on first run
                    if source_map is None:
                        b, t, _ = x.shape
                        source_map = torch.arange(t, device=x.device, dtype=torch.long).expand(b, -1)        
                    self._tome_info["source_map"] = merge_source_map(current_level_map, x, source_map)
                else: # 'matrix' mode
                    source_matrix = self._tome_info["source_matrix"]
                    self._tome_info["source_matrix"] = merge_source_matrix(merge, x, source_matrix)

            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class FPETAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as the metric
        metric = dict(
            metric = k.mean(1)
        )

        return x, metric

def make_fpet_class(transformer_class):
    class FPETVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["size"] = None
            self._tome_info["source_map"] = None
            self._tome_info["source_matrix"] = None

            return super().forward(*args, **kwdargs)

    return FPETVisionTransformer


"""
Faster Parameter-Efficient Tuning with Token Redundancy Reduction, CVPR'2025
    - paper (https://arxiv.org/abs/2503.20282)
    - code  (https://github.com/kyk120/fpet)
"""
def fpet_apply_patch(
    model: VisionTransformer, 
    trace_source: bool = True, 
    prop_attn: bool = True, 
    source_tracking_mode: str = 'map',
    method = 'none', 
    r_layer: int = 6, 
    dim_key = 64
):
    """
    Applies FPET to this transformer.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    FPETVisionTransformer = make_fpet_class(model.__class__)

    model.__class__ = FPETVisionTransformer
    model._tome_info = {
        "size": None,
        "source_map": None,      # For 'map' mode
        "source_matrix": None,   # For 'matrix' mode
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": getattr(getattr(model, 'module', model), 'cls_token', None) is not None,
        "distill_token": getattr(getattr(model, 'module', model), 'dist_token', None) is not None,
        "source_tracking_mode": source_tracking_mode,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    idx = 0
    for module in model.modules():
        if isinstance(module, (Block, TimmBlock)):
            if r_layer == idx:
                # TODO We do not support the Adapter in this version during inference
                # module.refinement = Adapter(dim=32, bit=1, in_dim=dim_key)
                if 'adaptformer' != method:
                    module.__class__ = FPETBlock
                    module._tome_info = model._tome_info
                else:
                    raise ValueError("[Warning] We do not support Adaptformer in this version.")
                    # bound_method = forward_block.__get__(module, module.__class__)
                    # setattr(module, 'forward', bound_method)
        elif isinstance(module, (Attention, TimmAttention)):
            if r_layer == idx:
                if 'lora' != method:
                    module.__class__ = FPETAttention
                else:
                    raise ValueError("[Warning] We do not support LoRA in this version.")
                    # bound_method = forward_attn.__get__(module, module.__class__)
                    # setattr(module, 'forward', bound_method)
        idx += 1
                