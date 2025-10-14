# /opentome/models/model.py

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.layers import trunc_normal_

from opentome.timm.tome import tome_apply_patch
from opentome.timm.dtem import dtem_apply_patch, trace_token_merge, token_unmerge_from_map_for_dtem
from opentome.tome.tome import token_unmerge_from_map, parse_r

class LocalEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 depth=4, feat_dim=None, window_size: int = None, r: int = 2, t: int = 1, num_classes=10):
        super().__init__()
        self.vit = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                                     depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=True, num_classes=0,
                                     drop_rate=0.0,attn_drop_rate=0.0,drop_path_rate=0.0,)

    def forward(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        # 重置跨 batch 的踪迹与屏蔽，避免状态泄漏
        self.vit._tome_info["token_map_for_dtem"] = None
        self.vit._tome_info["token_mask_for_dtem"] = None
        # 与 timm 的 ViT 对齐，启用 norm_pre
        x = self.vit.norm_pre(x)
        n = x.shape[1]
        self.vit._tome_info["r"] = parse_r(len(self.vit.blocks), self.vit.r, self.vit._tome_info.get("total_merge", None))
        self.vit._tome_info["size"] = torch.ones_like(x[..., 0:1])
        self.vit._tome_info["token_counts_local"] = []
        
        # 检查cls_token判断
        # has_cls_token = hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None
        # num_prefix_tokens = getattr(self.vit, 'num_prefix_tokens', 0)
        # print(f"[LocalEncoder] has_cls_token: {has_cls_token}, num_prefix_tokens: {num_prefix_tokens}")
        
        for i, blk in enumerate(self.vit.blocks):
            x, size, n, _ = blk(x, self.vit._tome_info["size"], n=n)
            self.vit._tome_info["token_counts_local"].append(x.shape[1])
        x = self.vit.norm(x)
        return x, self.vit._tome_info["size"], self.vit._tome_info


class LatentEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 depth=12, source_tracking_mode='map', prop_attn=True, window_size=None, use_naive_local=False, r: int = 2):
        super().__init__()
        self.vit = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                                     depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=True, num_classes=0,
                                     drop_rate=0.0,attn_drop_rate=0.0,drop_path_rate=0.0,)
        # 统一在 HybridToMeModel 中进行 apply_patch（去除未使用占位字段）

    def forward(self, x, size):
        # 重置跨 batch 的踪迹与屏蔽，避免状态泄漏
        self.vit._tome_info["token_map_for_dtem"] = None
        self.vit._tome_info["token_mask_for_dtem"] = None
        self.vit._tome_info["r"] = parse_r(len(self.vit.blocks), self.vit._tome_info["r"], self.vit._tome_info.get("total_merge", None))
        self.vit._tome_info["size"] = size
        self.vit._tome_info["source_map"] = None
        self.vit._tome_info["source_matrix"] = None
        self.vit._tome_info["token_counts_latent"] = []
        # print(f"self.vit._tome_info: {self.vit._tome_info}")
        
        # 检查cls_token判断
        # has_cls_token = hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None
        # num_prefix_tokens = getattr(self.vit, 'num_prefix_tokens', 0)
        # print(f"[LatentEncoder] has_cls_token: {has_cls_token}, num_prefix_tokens: {num_prefix_tokens}")
        
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            self.vit._tome_info["token_counts_latent"].append(x.shape[1])
            # print(f"blk._tome_info: {blk._tome_info}")
        x = self.vit.norm(x)
        return x, self.vit._tome_info["size"], self.vit._tome_info


class HybridToMeModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12,
                 mlp_ratio=4.0, dtem_feat_dim=None, local_depth=4, latent_depth=12,
                 tome_window_size=None, tome_use_naive_local=False, num_classes=10, dtem_window_size: int = None, dtem_r: int = 2, dtem_t: int = 1,
                 total_merge_local: int = 8, total_merge_latent: int = 4):
        super().__init__()
        self.local = LocalEncoder(img_size, patch_size, embed_dim, num_heads, mlp_ratio, num_classes=num_classes,
                                  depth=local_depth, feat_dim=dtem_feat_dim, window_size=dtem_window_size,
                                  r=total_merge_local//max(local_depth,1), t=dtem_t)
        self.latent = LatentEncoder(img_size, patch_size, embed_dim, num_heads, mlp_ratio,
                                    depth=latent_depth, source_tracking_mode='map',
                                    prop_attn=True, window_size=tome_window_size, use_naive_local=tome_use_naive_local,
                                    r=total_merge_latent//max(latent_depth,1)) if latent_depth>0 else None
        self.head = nn.Linear(embed_dim, num_classes)
        trunc_normal_(self.head.weight, std=.02)
        nn.init.zeros_(self.head.bias)
        # 统一 apply_patch
        self._apply_patches(dtem_feat_dim, dtem_window_size, dtem_t, total_merge_local, tome_window_size, tome_use_naive_local, total_merge_latent)

    def _apply_patches(self, dtem_feat_dim, dtem_window_size, dtem_t, total_merge_local, tome_window_size, tome_use_naive_local, total_merge_latent):
        # DTEM patch
        dtem_r_per_layer = total_merge_local//max(len(self.local.vit.blocks),1)
        dtem_apply_patch(self.local.vit, feat_dim=dtem_feat_dim, trace_source=True, prop_attn=True,
                         default_r=dtem_r_per_layer, window_size=dtem_window_size, t=dtem_t)
        # 记录总merge数，供 parse_r 使用
        self.local.vit._tome_info["total_merge"] = total_merge_local
        if self.latent is not None and len(self.latent.vit.blocks) > 0:
            tome_r_per_layer = total_merge_latent//max(len(self.latent.vit.blocks),1)
            from opentome.timm.tome import tome_apply_patch
            tome_apply_patch(self.latent.vit, trace_source=True, prop_attn=True, window_size=tome_window_size,
                                use_naive_local=tome_use_naive_local, r=tome_r_per_layer)
            self.latent.vit._tome_info["total_merge"] = total_merge_latent
    
    def forward_ori(self,x):
        x = self.local.forward(x)
        x = self.latent.forward(x[0], None)
        cls_token_repr = x[0][:, 0]
        logits = self.head(cls_token_repr)
        aux = {}
        return logits, aux

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        num_patches = self.local.vit.patch_embed.num_patches
        L_full = num_patches + self.local.vit.num_prefix_tokens

        # 阶段1：LocalEncoder（DTEM软合并 + 踪迹）
        x_local, size_local, info_local = self.local(x)
        token_map_for_dtem = info_local.get("token_map_for_dtem", None)
        if token_map_for_dtem is None:
            token_map_for_dtem = torch.arange(L_full, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        if self.training:
            # print("training")
            x_trace, size_trace, token_map_for_dtem_compact = trace_token_merge(
                x_local, size_local, token_map_for_dtem
            )
        else:
            # print("eval")
            x_trace, size_trace = x_local, size_local
            token_map_for_dtem_compact = token_map_for_dtem

        # 阶段3：LatentEncoder（ToMe硬合并）
        x_latent, size_latent, info_latent = self.latent(x_trace, size_trace)
        token_map_tome = info_latent.get("source_map", None)
        x_restore_tome = token_unmerge_from_map(x_latent, token_map_tome)

        # 阶段4：两阶段恢复（ToMe -> DTEM）
        x_out_full_seq = token_unmerge_from_map_for_dtem(
            x_restore_tome, token_map_for_dtem_compact, T_full=L_full
        )
        cls_token_repr = x_out_full_seq[:, 0]
        logits = self.head(cls_token_repr)

        aux = {"token_counts_local": info_local.get("token_counts_local", None)}
        return logits, aux
# python /yuchang/yk/benchmark_scaleup.py --devices cuda:0 --lengths 64000,128000,256000,512000,1024000,2048000,4096000 --num_workers 8 --model_name resnet50 --model_path /yuchang/yk/resnet50_mixup.pth