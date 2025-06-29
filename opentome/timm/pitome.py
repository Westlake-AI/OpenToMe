# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch
import torch.nn as nn
from timm.layers import Mlp, DropPath, use_fused_attn
from timm.models.vision_transformer import VisionTransformer, LayerScale
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock
from opentome.timm import Attention, Block
import math
from opentome.tome.tome import parse_r, merge_source, merge_wavg
from opentome.tome.pitome import pitome_vision


class PiToMeAttention(Attention):
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
            attn = attn +  size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, k.mean(1)
    

class PiToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def init_margin(self, margin=0.5):
        # self.margin = nn.Parameter(torch.tensor(margin)) 
        self.margin = margin

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn, metric = self.attn(self.norm1(x))
        x = x + self._drop_path1(x_attn)

        r = self._tome_info["r"].pop(0)
        use_bsm_pitome = self._tome_info["use_bsm_pitome"].pop(0)
        if r > 0:
            merge, _ = pitome_vision(
                                    metric,
                                    r,
                                    margin = self.margin,
                                    class_token = self._tome_info["class_token"],
                                    use_bsm_pitome = use_bsm_pitome
                                )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x,  self._tome_info["size"]  = merge_wavg(merge, x ,self._tome_info["size"]) 
        # print(r, x.shape, self.margin, use_bsm_pitome)
        
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x 



def make_tome_class(transformer_class):
    class PiToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            # self._tome_info["r"] = [self.ratio] * len(self.blocks) 
            # self._tome_info["use_bsm_pitome"] = [False] * (len(self.blocks)//2) + [True] * (len(self.blocks)//2)
            self._tome_info["r"] = parse_r(
                len(self.blocks), self.r, self._tome_info["total_merge"])
            num_bsm_layers = math.ceil(len(self.blocks) * 0.5) 
            self._tome_info["use_bsm_pitome"] = [True] * (num_bsm_layers) + [False] * (len(self.blocks) - num_bsm_layers)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)
            
    return PiToMeVisionTransformer



def pitome_apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = False):

    PiToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = PiToMeVisionTransformer
    model.r = 0 
    model._tome_info = {
        "ratio": model.r,
        "margin": [],
        "size": None,
        "source": None,
        "total_merge": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": getattr(model, 'cls_token', None) is not None,
        "distill_token": getattr(model, 'dist_token', None) is not None,
    }

    margins = [.75 - .75 * (i / len(model.blocks)) for i in range(len(model.blocks))]
    i = 0
    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, (Block, TimmBlock)):
            module.__class__ = PiToMeBlock
            module.init_margin(margins[i])
            module._tome_info = model._tome_info
            i += 1
        elif isinstance(module, (Attention, TimmAttention)):
            module.__class__ = PiToMeAttention