# model_ablation_no_sort.py
# 消融实验 A: 去除位置排序
# 对标: CLSHybridToMeModel (model.py)
# 变更: 移除 topk 后按 center_of_mass 的 argsort 排序过程
#       token 保持 topk 原始顺序（按 strength 降序），不依赖位置先验

import torch
import torch.nn as nn
from timm.models.registry import register_model

from opentome.models.mergenet.model import (
    CLSHybridToMeModel,
    MyCrossAttention,
)
from opentome.tome.tome import token_unmerge_from_map, parse_r


class AblationNoSortModel(CLSHybridToMeModel):
    """
    消融 A: 移除 center_of_mass 位置排序。

    与 CLSHybridToMeModel 唯一区别:
    topk 选择后不再按 center_of_mass 做 argsort，
    token 保持 topk 返回的原始顺序。
    """

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        num_patches = self.local.vit.patch_embed.num_patches
        L_full = num_patches + self.local.vit.num_prefix_tokens

        x_local, x_embed, size_local, info_local = self.local(x)
        source_matrix = info_local.get("source_matrix", None)

        k = L_full - info_local["total_merge"] - 1
        token_strength = size_local[..., 0]
        token_strength_no_cls = token_strength[:, 1:]
        if k <= 0 or k > token_strength_no_cls.shape[1]:
            k = token_strength_no_cls.shape[1]

        # ========== 消融核心: 只做 topk，不做 argsort ==========
        with torch.no_grad():
            topk_vals, topk_indices = torch.topk(
                token_strength_no_cls.detach(), k, dim=1,
                largest=True, sorted=True,   # sorted by strength descending
            )

        topk_x_trace = torch.gather(
            x_local, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, x_local.shape[-1]),
        )
        topk_size_trace = torch.gather(
            size_local, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, size_local.shape[-1]),
        )
        topk_x = torch.cat([x_local[:, :1], topk_x_trace], dim=1)
        topk_size = torch.cat(
            [size_local[:, :1, 0], topk_size_trace.squeeze(-1)], dim=-1,
        ).unsqueeze(-1)
        size_trace = topk_size

        # source_matrix bias (使用未排序的 topk_indices)
        if source_matrix is not None:
            with torch.no_grad():
                center = info_local["source_matrix_center"]
                width = info_local["source_matrix_width"]
                bias = torch.full((B, k + 1, L_full), -1e10, device=device, dtype=x_local.dtype)
                bias[:, 0, :] = 0.0

                actual_indices = topk_indices + 1
                source_for_topk = torch.gather(
                    source_matrix, 1,
                    actual_indices.unsqueeze(-1).expand(-1, -1, width),
                )
                offset_range = torch.arange(width, device=device).view(1, 1, -1)
                j_positions = actual_indices.unsqueeze(-1) + (offset_range - center)
                valid_mask = (j_positions >= 0) & (j_positions < L_full)
                log_source = torch.where(
                    source_for_topk > 1e-10,
                    torch.log(source_for_topk.clamp(min=1e-10)),
                    torch.full_like(source_for_topk, -1e10),
                )
                log_source_masked = torch.where(valid_mask, log_source, torch.full_like(log_source, -1e10))
                j_positions_safe = torch.where(valid_mask, j_positions, torch.zeros_like(j_positions))
                bias[:, 1:, :].scatter_(2, j_positions_safe, log_source_masked)
        else:
            bias = torch.zeros((B, k + 1, L_full), device=device, dtype=x_local.dtype)

        if self.total_merge_local == 0 and self.total_merge_latent == 0:
            x_trace = topk_x
        else:
            x_trace = self.encode_cross_attention(topk_x, x_embed, mask=bias) + topk_x

        x_latent, size_latent, info_latent = self.latent(x_trace, size_trace)
        cls_token_repr = x_latent[:, 0]
        logits = self.head(cls_token_repr)

        aux = {"token_counts_local": info_local.get("token_counts_local", None)}
        return logits, aux


@register_model
def ablation_nosort_small_cls(**kwargs):
    return AblationNoSortModel(arch='small', remove_decoder_cross_attention=True, **kwargs)
