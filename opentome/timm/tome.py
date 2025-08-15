#/zhoujingbo/yk/work/OpenToMe/opentome/timm/tome.py
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
# NOTE: Assumes the functions from the previous step are available in this module.
from opentome.tome.tome import (
    bipartite_soft_matching,
    merge_source_matrix,
    merge_source_map,
    merge_wavg,
    parse_r,
    # Assuming these functions were added to the tome.py file
    naive_local_bipartite_soft_matching,
    local_bipartite_soft_matching,
)
from opentome.timm import Attention, Block

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class ToMeAttention(Attention):
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


class ToMeBlock(Block):
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
            # --- NEW LOGIC: Select and apply the appropriate merge function ---
            h = self._tome_info.get("h")
            use_naive_local = self._tome_info.get("use_naive_local", False)
            
            metric_val = metric['metric']
            class_token = self._tome_info["class_token"]
            distill_token = self._tome_info["distill_token"]

            # If h is specified, use a local matching strategy
            if h is not None and h >= 0:
                if use_naive_local:
                    # Use the naive (but clear) local implementation
                    merge, _, current_level_map = naive_local_bipartite_soft_matching(
                        metric_val, r, h, class_token, distill_token
                    )
                else:
                    # Use the optimized local implementation
                    merge, _, current_level_map = local_bipartite_soft_matching(
                        metric_val, r, h, class_token, distill_token
                    )
            else:
                # If h is not specified, fall back to the original global matching
                merge, _, current_level_map = bipartite_soft_matching(
                    metric_val, r, class_token, distill_token
                )
            # --- END OF NEW LOGIC ---

            if self._tome_info["trace_source"]:
                # --- NEW: Select tracking strategy based on the config ---
                if self._tome_info["source_tracking_mode"] == 'map':
                    source_map = self._tome_info["source_map"]
                    
                    # Initialize map on first run
                    if source_map is None:
                        b, t, _ = x.shape
                        source_map = torch.arange(t, device=x.device, dtype=torch.long).expand(b, -1)
                        
                    self._tome_info["source_map"] = merge_source_map(
                        current_level_map, x, source_map
                    )
                else: # 'matrix' mode
                    source_matrix = self._tome_info["source_matrix"]
                    self._tome_info["source_matrix"] = merge_source_matrix(
                        merge, x, source_matrix
                    )

            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
         - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(
                len(self.module.blocks), self.r, self._tome_info["total_merge"])
            self._tome_info["size"] = None
            self._tome_info["source_map"] = None
            self._tome_info["source_matrix"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer



"""
Token Merging: Your ViT but Faster, ICLR'2023
    - paper (https://arxiv.org/abs/2210.09461)
    - code  (https://github.com/facebookresearch/ToMe)
"""
def tome_apply_patch(
    model: VisionTransformer,
    trace_source: bool = True,
    prop_attn: bool = True,
    h: Optional[int] = None,
    use_naive_local: bool = False,
    source_tracking_mode: str = 'map'
):
    print("\n✅ [dEbUg] Successfully called the MODIFIED tome_apply_patch function!\n")
    print(source_tracking_mode)
    
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.
    
    MODIFIED to support local token merging.

    Args:
        model (VisionTransformer): The model to apply ToMe to.
        trace_source (bool): If True, track the source of each token.
        prop_attn (bool): If True, apply proportional attention.
        h (Optional[int]): The locality parameter for merging. A token `a_i` can
            only be merged with a token `b_j` if `|i - j| <= h`. If `None` or a
            negative value, the original global merging is used.
        use_naive_local (bool): If `h` is specified, this flag determines the
            local merging implementation.
            - If `False` (default), uses the memory and computationally efficient
              `local_bipartite_soft_matching`.
            - If `True`, uses `naive_local_bipartite_soft_matching`, which is
              less efficient but easier to verify.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source_map": None,      # For 'map' mode
        "source_matrix": None,   # For 'matrix' mode
        "total_merge": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": getattr(getattr(model, 'module', model), 'cls_token', None) is not None,
        "distill_token": getattr(getattr(model, 'module', model), 'dist_token', None) is not None,
        "source_tracking_mode": source_tracking_mode,
        "h": h,
        "use_naive_local": use_naive_local,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, (Block, TimmBlock)):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, (Attention, TimmAttention)):
            module.__class__ = ToMeAttention