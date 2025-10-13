# ------ jinxin modified ------ #
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.layers import Mlp, DropPath, use_fused_attn
from timm.models.vision_transformer import VisionTransformer, LayerScale
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock

# --- MODIFIED: Import the new local matching functions ---
from opentome.tome.atc import agglomerative_clustering, atc_parse_r
from opentome.tome.tome import (
    merge_wavg,
)
from opentome.timm import Attention, Block

class ATCAttention(Attention):
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
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads
                                   ).permute(2, 0, 3, 1, 4))
        q, k, v = (
            qkv[0], qkv[1], qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        if self.fused_attn:  # pytorch flash-attn with ToMe
            x = F.scaled_dot_product_attention(q, k, v,
                attn_mask=None if size is None else size.log()[:, None, None, :, 0],
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:  # naive attn with ToMe
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if size is not None:  # Apply proportional attention
                attn = attn + size.log()[:, None, None, :, 0]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as the metric
        metric = dict(
            metric = k.mean(1)
        )

        return x, metric


class ATCBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
     - MODIFIED: Add logic to select between global and local merging strategies.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        assert isinstance(metric['metric'], (float, torch.Tensor)), "metric not a float or torch.Tensor"
        x = x + self._drop_path1(self.ls1(x_attn))
        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _, current_level_map = agglomerative_clustering(
                metric["metric"],
                r,
                self._tome_info["linkage"], 
                self._tome_info["class_token"], 
                self._tome_info["distill_token"]
            )
            # ------ ATC return source map by the cluster algom., we do not generate map ------ #
            if self._tome_info["trace_source"]:
                if self._tome_info["source_tracking_mode"] == 'map':    
                    self._tome_info["source_map"] = current_level_map
                else: # 'matrix' mode
                   raise ValueError("ATC do not support source matrix yet.")
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x
    


def make_atc_class(transformer_class):
    class ATCVisionTransformer(transformer_class):
        """
        Modifications:
         - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = atc_parse_r(len(self.blocks), self.r, self._tome_info["total_tokens"], \
                                            offcial=True, location=self._tome_info["reduction_loc"], \
                                            ratio=self._tome_info["ratio"]
                                    )
            self._tome_info["size"] = None
            self._tome_info["source_map"] = None
            self._tome_info["source_matrix"] = None

            return super().forward(*args, **kwdargs)

    return ATCVisionTransformer



"""
Agglomerative Token Clustering, ECCV'2024
    - paper (https://arxiv.org/pdf/2409.11923)
    - code  (https://github.com/JoakimHaurum/ATC)
"""
def atc_apply_patch(
    model: VisionTransformer,
    trace_source: bool = True,
    prop_attn: bool = True,
    global_red: bool = True,
    source_tracking_mode: str = 'map'
):    
    """
    Applies ATC to this transformer. Afterward, set r using model.r.
    
    MODIFIED to support local token merging.

    Args:
        model (VisionTransformer): The model to apply ToMe to.
        trace_source (bool): If True, track the source of each token.
        prop_attn (bool): If True, apply proportional attention.
    """
    ATCVisionTransformer = make_atc_class(model.__class__)

    model.__class__ = ATCVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source_map": None,      # For 'map' mode
        "source_matrix": None,   # For 'matrix' mode
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": getattr(getattr(model, 'module', model), 'cls_token', None) is not None,
        "distill_token": getattr(getattr(model, 'module', model), 'dist_token', None) is not None,
        "source_tracking_mode": source_tracking_mode,
        # ------ local params ------ #
        "reduction_loc": list[3, 6, 9],
        "linkage": str('average'),
        "global_red": global_red,
        "total_tokens": 197,
        "ratio": 0.5
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, (Block, TimmBlock)):
            module.__class__ = ATCBlock
            module._tome_info = model._tome_info
        elif isinstance(module, (Attention, TimmAttention)):
            module.__class__ = ATCAttention