# model_tome.py - ToME ablation model
# Replaces DTEM+Perceiver (soft differentiable merging + cross attention)
# with pure ToME bipartite matching (hard non-differentiable merging).

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.layers import trunc_normal_
from timm.models.registry import register_model

from opentome.tome.tome import (
    bipartite_soft_matching,
    merge_source_map,
    merge_wavg,
    parse_r,
    token_unmerge_from_map,
)
from opentome.timm.tome import tome_apply_patch
from opentome.timm.bias_local_attn import LocalBlock
from opentome.models.mergenet.model import LatentEncoder


class ToMELocalEncoder(nn.Module):
    """
    Local encoder with pure ToME bipartite matching for token compression.
    Structure: n LocalBlock transformer layers (semantic extraction)
               then n ToME hard merge steps (compression without understanding).

    Compared to the original LocalEncoder (DTEM):
      - No metric_layers (ToME uses token features directly as metric)
      - No DTEMMergeOnly block (replaced by bipartite_soft_matching)
      - No source_matrix tracking (uses simpler source_map)
      - No soft assignment (hard bipartite matching)
    """

    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 local_depth: int = 4, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 total_merge_local: int = 0, local_block_window: int = 16):
        super().__init__()

        if local_depth <= 0:
            raise ValueError("local_depth must be >= 1")

        self.local_depth = local_depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.total_merge_local = total_merge_local

        self.vit = VisionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=0, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=True, num_classes=0,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate
        )

        dpr = torch.linspace(0, drop_path_rate, local_depth).tolist()
        self.vit.blocks = nn.ModuleList([
            LocalBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                drop_path=dpr[i],
                local_window=local_block_window,
            )
            for i in range(local_depth)
        ])

        self.default_r = total_merge_local // max(local_depth, 1)

    def forward(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        x_layers = []
        for local_blk in self.vit.blocks:
            x = local_blk(x)
            x_layers.append(x)

        if not x_layers:
            raise RuntimeError("ToMELocalEncoder requires at least one local block.")

        x_embed = x_layers[-1]
        x_merge = x_embed

        r_list = parse_r(self.local_depth, self.default_r, self.total_merge_local)
        size = torch.ones_like(x_merge[..., 0:1])
        source_map = None
        token_counts = []

        for i in range(self.local_depth):
            r = r_list[i] if i < len(r_list) else 0
            if r <= 0:
                token_counts.append(x_merge.shape[1])
                continue

            merge, _, current_level_map = bipartite_soft_matching(
                x_merge, r, class_token=True, distill_token=False,
            )

            if source_map is None:
                b, t, _ = x_merge.shape
                source_map = torch.arange(t, device=x_merge.device, dtype=torch.long).expand(b, -1)
            source_map = merge_source_map(current_level_map, x_merge, source_map)

            x_merge, size = merge_wavg(merge, x_merge, size)
            token_counts.append(x_merge.shape[1])

        x_out = self.vit.norm(x_merge)

        info = {
            "source_map": source_map,
            "token_counts_local": token_counts,
            "total_merge": self.total_merge_local,
        }

        return x_out, x_embed, size, info


class ToMEHybridModel(nn.Module):
    """
    Ablation model replacing DTEM+Perceiver with pure ToME.

    Key differences from HybridToMeModel:
      - LocalEncoder uses ToME hard merge instead of DTEM soft merge
      - No cross attention (perceiver) since tokens are already compressed
      - No source_matrix / attention bias construction
      - No topk selection based on token strength
    """

    arch_zoo = {
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 768,
                         'local_depth': 4,
                         'latent_depth': 8,
                         'num_heads': 12,
                         'mlp_ratio': 4.0
                        }),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 384,
                         'local_depth': 4,
                         'latent_depth': 8,
                         'num_heads': 6,
                         'mlp_ratio': 4.0
                        }),
        **dict.fromkeys(['s_ext', 'small_extend'],
                        {'embed_dims': 384,
                         'local_depth': 4,
                         'latent_depth': 12,
                         'num_heads': 6,
                         'mlp_ratio': 4.0
                        }),
    }

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 tome_window_size=None,
                 tome_use_naive_local=False,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 num_classes=1000,
                 lambda_local: float = 2.0,
                 total_merge_latent: int = 4,
                 local_block_window: int = 16,
                 pretrained=None,
                 pretrained_type: str = 'vit',
                 load_full_pretrained: bool = True,
                 freeze_local_encoder: bool = False,
                 **kwargs):
        super().__init__()

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
            self.arch = arch.split("-")[0]
        else:
            raise ValueError("Wrong setups.")

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = self.arch_settings['embed_dims']
        self.num_heads = self.arch_settings['num_heads']
        self.mlp_ratio = self.arch_settings['mlp_ratio']
        self.local_depth = self.arch_settings['local_depth']
        self.latent_depth = self.arch_settings['latent_depth']

        num_patches = (img_size // patch_size) ** 2
        self.total_merge_local = int(num_patches * (lambda_local - 1) / lambda_local)
        self.lambda_local = lambda_local

        self.total_merge_latent = total_merge_latent
        self.tome_window_size = tome_window_size
        self.tome_use_naive_local = bool(tome_use_naive_local)
        self.local_block_window = local_block_window

        self.num_classes = num_classes

        self.local = ToMELocalEncoder(
            self.img_size,
            self.patch_size,
            self.embed_dim,
            self.num_heads,
            self.mlp_ratio,
            local_depth=self.local_depth,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            total_merge_local=self.total_merge_local,
            local_block_window=self.local_block_window,
        )

        self.latent = LatentEncoder(
            self.img_size, self.patch_size, self.embed_dim, self.num_heads, self.mlp_ratio,
            depth=self.latent_depth,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            source_tracking_mode='map', prop_attn=True,
            window_size=self.tome_window_size, use_naive_local=self.tome_use_naive_local,
            r=self.total_merge_latent // max(self.latent_depth, 1)
        ) if self.latent_depth > 0 else None

        self.head = nn.Linear(self.embed_dim, self.num_classes)
        trunc_normal_(self.head.weight, std=.02)
        nn.init.zeros_(self.head.bias)

        self._apply_patches()

        if pretrained:
            self._load_full_pretrained_weights(pretrained, img_size, pretrained_type, load_full_pretrained)

        if freeze_local_encoder:
            self.freeze_local_encoder()

    def _apply_patches(self):
        if self.latent is not None and len(self.latent.vit.blocks) > 0:
            tome_r_per_layer = self.total_merge_latent // max(len(self.latent.vit.blocks), 1)
            tome_apply_patch(
                self.latent.vit, trace_source=True, prop_attn=True,
                window_size=self.tome_window_size,
                use_naive_local=self.tome_use_naive_local,
                r=tome_r_per_layer,
            )
            self.latent.vit._tome_info["total_merge"] = self.total_merge_latent

    def _load_full_pretrained_weights(self, pretrained, img_size, pretrained_type='vit', load_full=True):
        import traceback
        from opentome.models.utils import load_pt_weights

        try:
            from timm.models import create_model

            if isinstance(pretrained, str):
                model_name = pretrained
            else:
                model_prefix = 'deit' if pretrained_type.lower() == 'deit' else 'vit'
                if self.embed_dim == 768 and self.num_heads == 12:
                    model_name = f'{model_prefix}_base_patch16_224'
                elif self.embed_dim == 384 and self.num_heads == 6:
                    model_name = f'{model_prefix}_small_patch16_224'
                elif self.embed_dim == 192 and self.num_heads == 3:
                    model_name = f'{model_prefix}_tiny_patch16_224'
                else:
                    print(f"[ToMEHybridModel] Warning: Cannot auto-determine pretrained model for "
                          f"embed_dim={self.embed_dim}, num_heads={self.num_heads}.")
                    return

            load_mode_str = "full" if load_full else "local only"
            print(f"[ToMEHybridModel] Loading {load_mode_str} pretrained weights from: {model_name}")

            pretrained_model = create_model(model_name, pretrained=True, img_size=img_size, num_classes=0)
            pretrained_state = pretrained_model.state_dict()

            print(f"[ToMEHybridModel] Loading blocks [0, {self.local_depth}) to Local Encoder...")
            load_pt_weights(
                target_vit=self.local.vit,
                pretrained_state=pretrained_state,
                start_block=0,
                end_block=self.local_depth,
                verbose=True,
            )

            if load_full and self.latent is not None and self.latent_depth > 0:
                total_pretrained_depth = self.local_depth + self.latent_depth
                print(f"[ToMEHybridModel] Loading blocks [{self.local_depth}, {total_pretrained_depth}) "
                      f"to Latent Encoder...")
                load_pt_weights(
                    target_vit=self.latent.vit,
                    pretrained_state=pretrained_state,
                    start_block=self.local_depth,
                    end_block=total_pretrained_depth,
                    verbose=True,
                )
            elif not load_full:
                print(f"[ToMEHybridModel] Skipping Latent Encoder weights (load_full=False)")

            print(f"[ToMEHybridModel] Successfully loaded {load_mode_str} pretrained weights")

        except Exception as e:
            print(f"[ToMEHybridModel] ERROR: Failed to load pretrained weights: {e}")
            traceback.print_exc()

    def freeze_local_encoder(self):
        print("[ToMEHybridModel] Freezing Local Encoder parameters...")
        frozen_params = 0
        total_params = 0
        for name, param in self.local.named_parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            total_params += param.numel()
        print(f"[ToMEHybridModel] Frozen {frozen_params:,} parameters in Local Encoder")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_model_params = sum(p.numel() for p in self.parameters())
        print(f"[ToMEHybridModel] Trainable: {trainable_params:,} / {total_model_params:,}")

    def unfreeze_local_encoder(self):
        print("[ToMEHybridModel] Unfreezing Local Encoder parameters...")
        for name, param in self.local.named_parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_model_params = sum(p.numel() for p in self.parameters())
        print(f"[ToMEHybridModel] Trainable: {trainable_params:,} / {total_model_params:,}")

    def forward(self, x):
        x_local, x_embed, size_local, info_local = self.local(x)

        if self.latent is not None:
            x_latent, size_latent, info_latent = self.latent(x_local, size_local)
            token_map = info_latent.get("source_map", None)
            x_restore = token_unmerge_from_map(x_latent, token_map)
        else:
            x_restore = x_local

        cls_token_repr = x_restore[:, 0]
        logits = self.head(cls_token_repr)

        aux = {"token_counts_local": info_local.get("token_counts_local", None)}
        return logits, aux


class CLSToMEHybridModel(ToMEHybridModel):
    """
    Classification variant: uses CLS token from LatentEncoder output directly.
    No decoder cross attention, no token unmerge needed.
    """

    def forward(self, x):
        x_local, x_embed, size_local, info_local = self.local(x)

        if self.latent is not None:
            x_latent, size_latent, info_latent = self.latent(x_local, size_local)
        else:
            x_latent = x_local

        cls_token_repr = x_latent[:, 0]
        logits = self.head(cls_token_repr)

        aux = {"token_counts_local": info_local.get("token_counts_local", None)}
        return logits, aux


# ------ Model Registration ------ #

@register_model
def tomevit_base(**kwargs):
    return ToMEHybridModel(arch='base', **kwargs)

@register_model
def tomevit_small(**kwargs):
    return ToMEHybridModel(arch='small', **kwargs)

@register_model
def tomevit_base_cls(**kwargs):
    return CLSToMEHybridModel(arch='base', **kwargs)

@register_model
def tomevit_small_cls(**kwargs):
    return CLSToMEHybridModel(arch='small', **kwargs)

@register_model
def tomevit_small_cls_ext(**kwargs):
    return CLSToMEHybridModel(arch='s_ext', **kwargs)


if __name__ == '__main__':
    from timm.models import create_model

    print("=" * 60)
    print("Creating tomevit_small_cls model...")

    model = create_model(
        'tomevit_small_cls',
        pretrained=False,
        num_classes=1000,
        img_size=224,
        patch_size=8,
        lambda_local=4.0,
        total_merge_latent=0,
        local_block_window=32,
        tome_window_size=32,
        tome_use_naive_local=False,
        freeze_local_encoder=False,
    )
    model.eval()
    print(model)

    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"\nInput shape: {dummy_input.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy_input = dummy_input.to(device)

    print("\n" + "=" * 60)
    print("Running forward pass...")

    with torch.no_grad():
        output = model(dummy_input)

    if isinstance(output, tuple):
        logits, aux = output
        print(f"\nOutput logits shape: {logits.shape}")
        print(f"Output logits (first 10): {logits[0, :10]}")
        print(f"\nAuxiliary info: {aux}")
    else:
        print(f"\nOutput shape: {output.shape}")
