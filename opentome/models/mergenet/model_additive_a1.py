# model_additive_a1.py — DTEM (learnable metric + soft merge) variant of A0
#
# Identical to A0 (CLSToMEHybridModel / tomevit_*_cls) EXCEPT:
#   - Local merge: DTEM (learnable metric_layers + DTEMMergeOnly soft merge)
#     instead of ToME hard bipartite matching
#   - Adds: metric_layers (one Linear per local layer) + DTEMMergeOnly block
#
# DTEM soft merge keeps ALL tokens (only adjusts weights), so we add a
# weight-based token selection step to match A0's output token count
# before passing to the LatentEncoder.

import torch
from timm.models.registry import register_model

from opentome.models.mergenet.model_a0 import CLSToMEHybridModel
from opentome.models.mergenet.model import LocalEncoder


class AdditiveDTEM_CLSModel(CLSToMEHybridModel):
    """
    DTEM ablation: identical to A0 (CLSToMEHybridModel), only the local
    token-compression mechanism is swapped from ToME bipartite matching to
    DTEM (learnable metric_layers + differentiable soft merge via DTEMMergeOnly).

    Key difference from ToME: DTEM soft merge keeps ALL tokens (only weights
    change). To match A0's latent-stage input size, forward() selects the
    top-k tokens by weight after soft merge, producing the same compressed
    token count as A0's hard merge.
    """

    def __init__(self, *args,
                 dtem_window_size=None,
                 dtem_t=1,
                 dtem_feat_dim=None,
                 use_softkmax=False,
                 swa_size=None,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
                 **kwargs):
        kwargs.pop('dtem_r', None)

        super().__init__(
            *args,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

        self.local = LocalEncoder(
            img_size=self.img_size, patch_size=self.patch_size,
            embed_dim=self.embed_dim, num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            local_depth=self.local_depth,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            dtem_feat_dim=dtem_feat_dim,
            dtem_window_size=dtem_window_size,
            dtem_t=dtem_t,
            total_merge_local=self.total_merge_local,
            use_softkmax=use_softkmax,
            swa_size=swa_size,
            local_block_window=self.local_block_window,
        )

    def forward(self, x):
        x_local, x_embed, size_local, info_local = self.local(x)
        # x_local: [B, N_full, C]  (DTEM soft merge keeps ALL tokens)
        # size_local: [B, N_full, 1] (token weights after soft merge)

        # DTEM soft merge does not reduce token count — select top-k tokens
        # by weight to match A0's compressed output (num_patches/lambda + 1).
        num_patches = (self.img_size // self.patch_size) ** 2
        k = num_patches - self.total_merge_local  # same count as A0 after hard merge

        cls_token = x_local[:, :1]       # [B, 1, C]
        cls_size = size_local[:, :1]     # [B, 1, 1]
        patch_tokens = x_local[:, 1:]    # [B, N_patches, C]
        patch_size = size_local[:, 1:]   # [B, N_patches, 1]

        if k > 0 and k < patch_tokens.shape[1]:
            weights = patch_size[..., 0]  # [B, N_patches]
            with torch.no_grad():
                _, topk_idx = torch.topk(weights, k, dim=1, sorted=False)
                topk_idx, _ = torch.sort(topk_idx, dim=1)

            selected_tokens = torch.gather(
                patch_tokens, 1,
                topk_idx.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1]),
            )
            selected_size = torch.gather(
                patch_size, 1,
                topk_idx.unsqueeze(-1).expand(-1, -1, patch_size.shape[-1]),
            )
            x_compressed = torch.cat([cls_token, selected_tokens], dim=1)
            size_compressed = torch.cat([cls_size, selected_size], dim=1)
        else:
            x_compressed = x_local
            size_compressed = size_local

        if self.latent is not None:
            x_latent, size_latent, info_latent = self.latent(
                x_compressed, size_compressed,
            )
        else:
            x_latent = x_compressed

        cls_token_repr = x_latent[:, 0]
        logits = self.head(cls_token_repr)

        aux = {"token_counts_local": info_local.get("token_counts_local", None)}
        return logits, aux


# -------------------- Model Registration -------------------- #

@register_model
def additive_dtem_small_cls(**kwargs):
    return AdditiveDTEM_CLSModel(arch='small', **kwargs)


@register_model
def additive_dtem_base_cls(**kwargs):
    return AdditiveDTEM_CLSModel(arch='base', **kwargs)


@register_model
def additive_dtem_small_cls_ext(**kwargs):
    return AdditiveDTEM_CLSModel(arch='s_ext', **kwargs)


# -------------------- Smoke Test -------------------- #

if __name__ == '__main__':
    from timm.models import create_model

    cfg = dict(
        pretrained=False,
        num_classes=100,
        img_size=224,
        patch_size=16,
        lambda_local=4.0,
        total_merge_latent=0,
        local_block_window=16,
        dtem_window_size=7,
        dtem_t=1,
    )

    print("=" * 60)
    print("DTEM LocalEncoder: additive_dtem_small_cls")
    model = create_model('additive_dtem_small_cls', **cfg)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad():
        logits, aux = model(dummy)

    print(f"  logits shape : {logits.shape}")
    print(f"  aux          : {aux}")
    assert logits.shape == (2, 100), f"Unexpected shape {logits.shape}"

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  total params : {total_p:,}")

    metric_p = sum(p.numel() for p in model.local.metric_layers.parameters())
    print(f"  metric_layers params: {metric_p:,}")
    print("  PASS")
