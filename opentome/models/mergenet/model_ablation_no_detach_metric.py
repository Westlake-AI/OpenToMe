# model_ablation_no_detach_metric.py
# 消融实验 D: 去除 metric_layers 输入的 .detach()
# 对标: 原 DTEM 论文设计（解耦 metric 学习）
# 变更: 允许分类 loss 的梯度反向传播到 metric_layers 的输入，实现端到端学习

import torch
import torch.nn as nn
from timm.models.registry import register_model

from opentome.models.mergenet.model import (
    CLSHybridToMeModel,
    LocalEncoder,
    DTEMMergeOnly,
)
from opentome.tome.tome import parse_r


class LocalEncoderNoDetach(LocalEncoder):
    """
    消融 D: 移除 metric_layers 输入的 .detach()

    原始代码: metric = self.metric_layers[i](x_metric.detach())
    消融代码: metric = self.metric_layers[i](x_metric)

    允许分类 loss 梯度经由 metric → merge 权重 → 反传到 backbone 特征。
    """

    def forward(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        n = x.shape[1]
        x_layers = []
        for local_blk in self.vit.blocks:
            x = local_blk(x)
            x_layers.append(x)
        if not x_layers:
            raise RuntimeError("LocalEncoder requires at least one local block.")
        x_embed = x_layers[-1]
        x_merge = x_embed
        r_list = parse_r(
            self.local_depth,
            self.default_r,
            self._tome_info.get("total_merge", None),
        )
        self._tome_info["r"] = r_list
        self._tome_info["size"] = torch.ones_like(x[..., 0:1])
        self._prepare_trace_for_forward()
        self._tome_info["token_counts_local"] = []

        size = self._tome_info["size"]
        source_matrix = None

        for i, layer_x in enumerate(x_layers):
            x_metric = self._aggregate_with_source_matrix(layer_x, size, source_matrix)
            # ========== 消融核心: 移除 .detach() ==========
            metric = self.metric_layers[i](x_metric)
            r = r_list[i] if i < len(r_list) else 0

            x_merge, size, n, _, source_matrix = self.merge_block._merge_train(
                x_merge, size, r, n, {"metric": metric}, source_matrix
            )

            self._tome_info["size"] = size
            self._tome_info["token_counts_local"].append(x_merge.shape[1])

        x_out = self.vit.norm(x_merge)
        self._finalize_trace_for_forward(source_matrix)
        return x_out, x_embed, self._tome_info["size"], self._tome_info


class AblationNoDetachMetricModel(CLSHybridToMeModel):
    """
    消融 D: 允许 metric_layers 的梯度端到端流动。

    使用 LocalEncoderNoDetach 替换 LocalEncoder，
    其余结构与 CLSHybridToMeModel 完全相同。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old_local = self.local
        self.local = LocalEncoderNoDetach(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            local_depth=self.local_depth,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            dtem_feat_dim=self.dtem_feat_dim,
            dtem_window_size=self.dtem_window_size,
            dtem_t=self.dtem_t,
            total_merge_local=self.total_merge_local,
            use_softkmax=self.use_softkmax,
            swa_size=self.swa_size,
            local_block_window=self.local_block_window,
            source_trace_mode=self.source_trace_mode,
        )
        # 复制已加载的权重
        self.local.load_state_dict(old_local.state_dict(), strict=False)
        # 同步 _tome_info
        if hasattr(self, '_apply_patches'):
            self._apply_patches(
                self.dtem_feat_dim, self.dtem_window_size, self.dtem_t,
                self.total_merge_local, self.tome_window_size,
                self.tome_use_naive_local, self.total_merge_latent,
                self.use_softkmax, self.swa_size,
            )


@register_model
def ablation_nodetach_small_cls(**kwargs):
    return AblationNoDetachMetricModel(arch='small', remove_decoder_cross_attention=True, **kwargs)
