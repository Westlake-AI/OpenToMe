# model_ablation_no_crossattn.py
# 消融实验 B: 去除 Cross Attention (Perceiver)
# 对标: CLSToMEHybridModel (model_tome.py) — ToMe 也没有 cross attention
# 变更: topk 选择后直接送入 LatentEncoder，不经过 encode_cross_attention

import torch
import torch.nn as nn
from timm.models.registry import register_model

from opentome.models.mergenet.model import CLSHybridToMeModel


class AblationNoCrossAttnModel(CLSHybridToMeModel):
    """
    消融 B: 移除 encode_cross_attention。

    topk 选择后直接送入 LatentEncoder，
    不做 Perceiver-style cross attention，不构建 source_matrix bias。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'encode_cross_attention'):
            del self.encode_cross_attention

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        num_patches = self.local.vit.patch_embed.num_patches
        L_full = num_patches + self.local.vit.num_prefix_tokens

        x_local, x_embed, size_local, info_local = self.local(x)

        k = L_full - info_local["total_merge"] - 1
        token_strength = size_local[..., 0]
        token_strength_no_cls = token_strength[:, 1:]
        if k <= 0 or k > token_strength_no_cls.shape[1]:
            k = token_strength_no_cls.shape[1]

        with torch.no_grad():
            topk_vals, topk_indices = torch.topk(
                token_strength_no_cls.detach(), k, dim=1,
                largest=True, sorted=False,
            )
            # 按原始索引排序以保持空间顺序
            topk_indices_sorted, _ = torch.sort(topk_indices, dim=1)

        topk_x_trace = torch.gather(
            x_local, 1,
            topk_indices_sorted.unsqueeze(-1).expand(-1, -1, x_local.shape[-1]),
        )
        topk_size_trace = torch.gather(
            size_local, 1,
            topk_indices_sorted.unsqueeze(-1).expand(-1, -1, size_local.shape[-1]),
        )
        topk_x = torch.cat([x_local[:, :1], topk_x_trace], dim=1)
        topk_size = torch.cat(
            [size_local[:, :1, 0], topk_size_trace.squeeze(-1)], dim=-1,
        ).unsqueeze(-1)

        # 直接送入 latent (无 cross attention)
        x_latent, size_latent, info_latent = self.latent(topk_x, topk_size)
        cls_token_repr = x_latent[:, 0]
        logits = self.head(cls_token_repr)

        aux = {"token_counts_local": info_local.get("token_counts_local", None)}
        return logits, aux


@register_model
def ablation_noxattn_small_cls(**kwargs):
    return AblationNoCrossAttnModel(arch='small', remove_decoder_cross_attention=True, **kwargs)
