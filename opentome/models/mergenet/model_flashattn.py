# /opentome/models/model_flashattn.py

import torch
import torch.nn as nn
import math
from typing import Optional

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: Flash Attention not available. Falling back to standard attention.")

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

class FlashAttention(nn.Module):
    """Flash/standard attention with optional local window (h)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., h: Optional[int] = None):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.h = h

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _local_attn_mask(self, N: int, h: int, device: torch.device) -> torch.Tensor:
        # band mask: True where masked (out of window)
        i = torch.arange(N, device=device).view(-1, 1)
        j = torch.arange(N, device=device).view(1, -1)
        return (j - i).abs() > h

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # (B, H, N, D)

        use_local = self.h is not None and self.h > 0

        if use_local:
            if FLASH_ATTN_AVAILABLE and q.dtype in (torch.float16, torch.bfloat16):
                # Flash local window attention: expects (B, N, H, D)
                q_nhwd = (q * self.scale).permute(0, 2, 1, 3).contiguous()
                k_nhwd = k.permute(0, 2, 1, 3).contiguous()
                v_nhwd = v.permute(0, 2, 1, 3).contiguous()
                out = flash_attn_func(
                    q_nhwd, k_nhwd, v_nhwd,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    softmax_scale=1.0,
                    causal=False,
                    window_size=(self.h, self.h),
                    deterministic=True,
                )
                if out.ndim == 3 and out.shape[-1] == C:
                    attn_output = out
                else:
                    # (B, N, H, D) -> (B, N, C)
                    attn_output = out.reshape(B, N, C)
            else:
                # Fallback: standard attention with band mask
                attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
                mask = self._local_attn_mask(N, self.h, attn.device)  # (N, N)
                attn = attn.masked_fill(mask, float('-inf'))
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                out_bhnd = attn @ v
                attn_output = out_bhnd.transpose(1, 2).reshape(B, N, C)
        else:
            if FLASH_ATTN_AVAILABLE and self.training and q.dtype in (torch.float16, torch.bfloat16):
                # Global flash attention during training
                out = flash_attn_func(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
                attn_output = out.transpose(1, 2).reshape(B, N, C)
            else:
                # Standard global attention
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                out_bhnd = attn @ v
                attn_output = out_bhnd.transpose(1, 2).reshape(B, N, C)

        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        return attn_output


class FlashAttentionBlock(nn.Module):
    """Standard Transformer Block with Flash Attention (supports local window)."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h: int | None = None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FlashAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, h=h)
        # NOTE: drop_path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FlashAttentionEncoder(nn.Module):
    """Standard Vision Transformer Encoder with Flash Attention (supports local window)."""

    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 depth=16, qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., h: int | None = None):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        self.pre_norm = nn.LayerNorm(embed_dim)
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            FlashAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], h=h)
            for i in range(depth)])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.pre_norm(x)
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x


class FlashAttentionModel(nn.Module):
    """Standard Vision Transformer with Flash Attention for comparison (supports local window)."""

    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12,
                 mlp_ratio=4.0, depth=16, num_classes=10, qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., h: int | None = None):
        super().__init__()
        
        self.encoder = FlashAttentionEncoder(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, depth=depth,
            qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, h=h
        )
        
        self.head = nn.Linear(embed_dim, num_classes)
        trunc_normal_(self.head.weight, std=.02)
        nn.init.zeros_(self.head.bias)
        
    def forward(self, x):
        x = self.encoder(x)
        cls_token_repr = x[:, 0]  # Extract CLS token
        logits = self.head(cls_token_repr)
        aux = {}
        return logits, aux


def create_flash_attention_model(img_size=224, patch_size=16, embed_dim=768, num_heads=12,
                               mlp_ratio=4.0, depth=16, num_classes=10, h: int | None = None, **kwargs):
    """Factory function to create Flash Attention model"""
    return FlashAttentionModel(
        img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
        num_heads=num_heads, mlp_ratio=mlp_ratio, depth=depth, num_classes=num_classes,
        h=h, **kwargs
    )


# Convenience functions for different model sizes
def flash_attention_tiny(num_classes=10, **kwargs):
    return create_flash_attention_model(
        embed_dim=192, num_heads=3, depth=12, num_classes=num_classes, **kwargs
    )

def flash_attention_small(num_classes=10, **kwargs):
    return create_flash_attention_model(
        embed_dim=384, num_heads=6, depth=12, num_classes=num_classes, **kwargs
    )

def flash_attention_base(num_classes=10, **kwargs):
    return create_flash_attention_model(
        embed_dim=768, num_heads=12, depth=12, num_classes=num_classes, **kwargs
    )

def flash_attention_large(num_classes=10, **kwargs):
    return create_flash_attention_model(
        embed_dim=1024, num_heads=16, depth=24, num_classes=num_classes, **kwargs
    )

def flash_attention_huge(num_classes=10, **kwargs):
    return create_flash_attention_model(
        embed_dim=1280, num_heads=16, depth=32, num_classes=num_classes, **kwargs
    )
