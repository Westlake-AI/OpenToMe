# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

# ------ jinxin modified ------ #
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.layers import Mlp, DropPath, use_fused_attn
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock

from opentome.tome.tome import parse_r
from opentome.tome.dct import dc_matching
from opentome.timm import Attention, Block

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class DCTBlock(Block):
    """
    Modifications:
     - Apply DCT between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        x_attn = self.attn(self.norm1(x))
        x = x + self._drop_path1(x_attn)
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        r = self._tome_info["r"].pop(0)
        if r > 0:
            x = dc_matching(
                    x, 
                    r, 
                    class_token=self._tome_info["class_token"],
                    distill_token=self._tome_info["distill_token"],
                )
        return x


def make_tome_class(transformer_class):
    class DCTVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:            
            self._tome_info["r"] = parse_r(
                len(self.blocks), self.r, self._tome_info["total_merge"])
            self._tome_info["size"] = None
            self._tome_info["source_map"] = None
            self._tome_info["source_matrix"] = None

            return super().forward(*args, **kwdargs)

    return DCTVisionTransformer



"""
Fourier Transformer: Fast Long Range Modeling by Removing Sequence Redundancy with FFT Operator, ACL'2023
    - paper (https://arxiv.org/abs/2305.15099)
    - code  (https://github.com/LUMIA-Group/FourierTransformer)
"""
def dct_apply_patch(
    model: VisionTransformer, 
    trace_source: bool = True, 
    prop_attn: bool = True,
    source_tracking_mode: str = 'matrix'
):

    DCTVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = DCTVisionTransformer
    model.r = 0
    # model.cls_token = getattr(model, 'cls_token', None)
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source_map": None,      # For 'map' mode
        "source_matrix": None,   # For 'matrix' mode
        "total_merge": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": getattr(model, 'cls_token', None) is not None,
        "distill_token": getattr(model, 'dist_token', None) is not None,
        "source_tracking_mode": source_tracking_mode,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, (Block, TimmBlock)):
            module.__class__ = DCTBlock
            module._tome_info = model._tome_info
