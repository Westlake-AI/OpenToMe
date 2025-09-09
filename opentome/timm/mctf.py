# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.layers import Mlp, DropPath, use_fused_attn
from timm.models.vision_transformer import VisionTransformer, LayerScale
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock

from opentome.tome.tome import (
    merge_source_matrix,
    merge_source_map,
    parse_r,
)
from opentome.tome.mctf import mctf_bipartite_soft_matching, mctf_merge_wavg
from opentome.timm import Attention, Block

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class MCTFAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None, return_attn: bool = False
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
        attn_ = attn.clone()

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        metric = dict(
            metric = k.mean(1),
            attn = attn_,
        )

        return x, metric

        # if return_attn:
        #     return x, attn_
        # else:
        #     return x, k.mean(dim=1)


class MCTFBlock(Block):
    """
    Modifications:
     - Apply MCTF between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_ = x.clone()
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size, return_attn=self._tome_info["return_attn"])   # In MCTF, we return the attention metric, not k.mean(dim=1)
        assert isinstance(metric['metric'], (float, torch.Tensor)), "metric not a float or torch.Tensor"
        assert isinstance(metric['attn'], (float, torch.Tensor)), "attn not a float or torch.Tensor"
        x = x + self._drop_path1(x_attn)
        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _, current_level_map = mctf_bipartite_soft_matching(
                            x_,
                            r,
                            class_token = self._tome_info["class_token"],
                            distill_token = self._tome_info["distill_token"],
                            attn = metric["attn"] if self._tome_info["return_attn"] else metric["metric"],
                            tau_sim = self._tome_info["tau_sim"],
                            tau_info = self._tome_info["tau_info"],
                            tau_size = self._tome_info["tau_size"],
                            size = self._tome_info["size"],
                            bidirection = self._tome_info["bidirection"]
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
            
            x, self._tome_info["size"], _ = mctf_merge_wavg(
                                                merge, 
                                                x, 
                                                size=self._tome_info["size"], 
                                                metric = metric["attn"] if self._tome_info["return_attn"] else metric["k_mean"],
                                                one_step_ahead=self._tome_info["one_step_ahead"],
                                                pooling_type=self._tome_info["pooling_type"]
                                            )
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


def make_tome_class(transformer_class):
    class MCTFVisionTransformer(transformer_class):
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

    return MCTFVisionTransformer


"""
Multi-criteria Token Fusion with One-step-ahead Attention for Efficient Vision Transformers, CVPR'2024
    - paper (https://arxiv.org/abs/2403.10030)
    - code  (https://github.com/mlvlab/MCTF)
"""
def mctf_apply_patch(
    model: VisionTransformer, 
    trace_source: bool = True, 
    prop_attn: bool = True,
    source_tracking_mode: str = 'map'
):
    """ Apply MCTF patch to a VisionTransformer model. 
        This modifies the model's class to MCTFVisionTransformer and updates the blocks and attention layers to MCTFBlock 
        and MCTFAttention respectively.
        Args:
            model (VisionTransformer): The model to apply the patch to.
            trace_source (bool): Whether to trace the source of tokens during merging. Defaults to True.
            prop_attn (bool): Whether to propagate attention information. Defaults to True.
        Returns:
            None: The model is modified in place.
    """
    MCTFVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = MCTFVisionTransformer
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
        # MCFT hyperparameters
        "one_step_ahead": 1,
        "tau_sim": 1,
        "tau_info": 0,
        "tau_size": 0,
        "bidirection": True,
        "pooling_type": 'none', # ['none', 'max', 'mean]
        "return_attn": True,    # TODO need to support k.mean(1) as the metric, not only attention
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, (Block, TimmBlock)):
            module.__class__ = MCTFBlock
            module._tome_info = model._tome_info
        elif isinstance(module, (Attention, TimmAttention)):
            module.__class__ = MCTFAttention
