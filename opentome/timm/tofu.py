import torch
from typing import Tuple
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock

from opentome.tome.tome import bipartite_soft_matching, merge_source, merge_wavg, parse_r
from opentome.timm import Attention, Block


class ToFuAttention(Attention):
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

        metric = dict(
            metric = k.mean(1)
        )

        return x, metric


class ToFuBlock(Block):
    """
    Modifications:
     - Apply ToFu between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def init_strategy(self, strategy='mean'):
        self.strategy = strategy 


    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        assert isinstance(metric['metric'], (float, torch.Tensor)), "metric not a float or torch.Tensor"
        x = x + self._drop_path1(x_attn)
        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric['metric'],
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(merge, x, self._tome_info["source"])
                
            x = merge(x, mode=self.strategy)

        # print(r, x.shape, self.strategy)

        x = x + self._drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def make_tofme_class(transformer_class):
    class ToFuVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(
                len(self.blocks), self.r, self._tome_info["total_merge"])
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToFuVisionTransformer


"""
Token Fusion: Bridging the Gap between Token Pruning and Token Merging, WACV'2024
    - paper (https://arxiv.org/abs/2312.01026)
"""
def tofu_apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):

    ToFuVisionTransformer = make_tofme_class(model.__class__)

    model.__class__ = ToFuVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "total_merge": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": getattr(model, 'cls_token', None) is not None,
        "distill_token": getattr(model, 'dist_token', None) is not None,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    strategies = ['tofu' if i > len(model.blocks) // 2 else 'prune' for i in range(len(model.blocks))]
    current_layer = 0
    for module in model.modules():
        if isinstance(module, (Block, TimmBlock)):
            module.__class__ = ToFuBlock
            module.init_strategy(strategies[current_layer])
            module._tome_info = model._tome_info
            current_layer += 1
        elif isinstance(module, (Attention, TimmAttention)):
            module.__class__ = ToFuAttention
            module._tome_info = model._tome_info