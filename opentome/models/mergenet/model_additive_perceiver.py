# model_additive_perceiver.py — Cross-Attention Pooling (Perceiver Resampling) LocalEncoder
#
# Identical to A0 (CLSToMEHybridModel / tomevit_*_cls) EXCEPT:
#   - Local merge: Perceiver Resampling (learnable latent queries + cross-attention)
#     instead of ToME hard bipartite matching
#   - Adds: learnable latent queries + PerceiverResamplingBlock(s)
#
# Output token count matches ToME: num_patches / lambda_local + 1 (CLS).
# Everything else (LocalBlock layers, LatentEncoder, head, _apply_patches)
# is shared with A0 via inheritance from CLSToMEHybridModel.

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.layers import trunc_normal_

from opentome.timm.bias_local_attn import LocalBlock
from opentome.models.mergenet.model_a0 import CLSToMEHybridModel


class PerceiverResamplingBlock(nn.Module):
    """Pre-norm cross-attention + FFN block (Perceiver Resampler style)."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_drop, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, queries, kv):
        q = self.norm_q(queries)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q, kv_n, kv_n)
        queries = queries + attn_out
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class PerceiverLocalEncoder(nn.Module):
    """
    Drop-in replacement for ToMELocalEncoder.
    Same LocalBlock layers for feature extraction, but compresses tokens via
    Perceiver Resampling (cross-attention pooling) instead of bipartite merge.

    Output shape matches ToME: [CLS] + K latent queries, where
    K = num_patches - total_merge_local = num_patches / lambda_local.
    """

    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12,
                 mlp_ratio=4.0, local_depth: int = 4, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.0,
                 total_merge_local: int = 0, local_block_window: int = 16,
                 perceiver_num_layers: int = 1):
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
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        dpr = torch.linspace(0, drop_path_rate, local_depth).tolist()
        self.vit.blocks = nn.ModuleList([
            LocalBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, attn_drop=attn_drop_rate, proj_drop=drop_rate,
                drop_path=dpr[i], local_window=local_block_window,
            )
            for i in range(local_depth)
        ])

        num_patches = (img_size // patch_size) ** 2
        self.num_latent_tokens = max(num_patches - total_merge_local, 1)

        self.latent_queries = nn.Parameter(
            torch.zeros(1, self.num_latent_tokens, embed_dim)
        )
        trunc_normal_(self.latent_queries, std=0.02)

        self.perceiver_layers = nn.ModuleList([
            PerceiverResamplingBlock(
                embed_dim, num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate,
            )
            for _ in range(perceiver_num_layers)
        ])

    def forward(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        for blk in self.vit.blocks:
            x = blk(x)

        x_embed = x

        cls_token = x[:, :1, :]
        queries = self.latent_queries.expand(x.size(0), -1, -1)
        for layer in self.perceiver_layers:
            queries = layer(queries, x)

        x_out = torch.cat([cls_token, queries], dim=1)
        x_out = self.vit.norm(x_out)

        size = torch.ones_like(x_out[..., 0:1])
        info = {
            "source_map": None,
            "token_counts_local": [x_out.shape[1]],
            "total_merge": self.total_merge_local,
        }
        return x_out, x_embed, size, info


class AdditivePerceiver_CLSModel(CLSToMEHybridModel):
    """
    Perceiver ablation: identical to A0 (CLSToMEHybridModel), only the local
    token-compression mechanism is swapped from ToME bipartite matching to
    Perceiver Resampling (cross-attention pooling with learnable latent queries).
    """

    def __init__(self, *args,
                 perceiver_num_layers=1,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
                 **kwargs):
        kwargs.pop('dtem_window_size', None)
        kwargs.pop('dtem_r', None)
        kwargs.pop('dtem_t', None)
        kwargs.pop('dtem_feat_dim', None)
        kwargs.pop('use_softkmax', None)
        kwargs.pop('swa_size', None)

        super().__init__(
            *args,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

        self.local = PerceiverLocalEncoder(
            img_size=self.img_size, patch_size=self.patch_size,
            embed_dim=self.embed_dim, num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            local_depth=self.local_depth,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            total_merge_local=self.total_merge_local,
            local_block_window=self.local_block_window,
            perceiver_num_layers=perceiver_num_layers,
        )


# -------------------- Model Registration -------------------- #

@register_model
def additive_perceiver_small_cls(**kwargs):
    return AdditivePerceiver_CLSModel(arch='small', **kwargs)


@register_model
def additive_perceiver_base_cls(**kwargs):
    return AdditivePerceiver_CLSModel(arch='base', **kwargs)


@register_model
def additive_perceiver_small_cls_ext(**kwargs):
    return AdditivePerceiver_CLSModel(arch='s_ext', **kwargs)


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
    )

    print("=" * 60)
    print("Perceiver Resampling: additive_perceiver_small_cls")
    model = create_model('additive_perceiver_small_cls', **cfg)
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

    perceiver_p = sum(p.numel() for p in model.local.perceiver_layers.parameters())
    query_p = model.local.latent_queries.numel()
    print(f"  perceiver params (cross-attn + FFN): {perceiver_p:,}")
    print(f"  latent query params: {query_p:,}")
    print("  PASS")
