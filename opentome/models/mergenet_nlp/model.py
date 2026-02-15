
from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from fla.layers.attn import Attention
from fla.modules import GatedMLP as TransformerMLP
from fla.models.utils import Cache

# Use fla's fused RMSNorm if CUDA is available (faster), otherwise fallback to PyTorch native
try:
    from fla.modules import RMSNorm as FLARMSNorm
    _use_fla_rmsnorm = torch.cuda.is_available()
except ImportError:
    _use_fla_rmsnorm = False

def get_rmsnorm(hidden_size, eps=1e-6):
    """Get RMSNorm implementation based on device availability."""
    if _use_fla_rmsnorm:
        return FLARMSNorm(hidden_size, eps=eps)
    else:
        # Fallback to PyTorch native (torch 2.4+)
        return nn.RMSNorm(hidden_size, eps=eps)
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules.l2warp import l2_warp

from opentome.models.modeling_layers import GradientCheckpointingLayer
from opentome.timm.dtem import DTEMBlock
from opentome.timm.bias_local_attn import LocalBlock
from opentome.tome.tome import parse_r

from .configuration_mergenet import MergeNetConfig

logger = logging.get_logger(__name__)


# =====================================================================
# Cross-Attention Module (reused from CV version)
# =====================================================================

class MyCrossAttention(nn.Module):
    """
    Multi-head cross attention where query and key/value sequences
    can have different lengths and batch sizes.
    """

    def __init__(self, embed_dim, num_heads=8, bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, mask=None):
        """
        Args:
            q: (Bq, Nq, C) -- queries
            kv: (Bk, Nk, C) -- keys/values
            mask: (Bq, Nq, Nk), optional -- attention bias (additive)
        Returns:
            context: (Bq, Nq, C)
        """
        Bq, Nq, C = q.shape
        Bk, Nk, Ck = kv.shape
        assert C == self.embed_dim and Ck == self.embed_dim

        # Compute projections
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(kv)
        v_proj = self.v_proj(kv)

        # Reshape for multi-head
        q_proj = q_proj.reshape(Bq, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = k_proj.reshape(Bk, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v_proj = v_proj.reshape(Bk, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle batch broadcasting
        if Bq != Bk:
            if Bq == 1:
                q_proj = q_proj.expand(Bk, -1, -1, -1)
                B = Bk
            elif Bk == 1:
                k_proj = k_proj.expand(Bq, -1, -1, -1)
                v_proj = v_proj.expand(Bq, -1, -1, -1)
                B = Bq
            else:
                raise ValueError(f"Incompatible batch sizes: q {Bq}, kv {Bk}")
        else:
            B = Bq

        # Attention scores (compute in float32 for numerical stability)
        attn_scores = torch.matmul(
            q_proj.float(), k_proj.float().transpose(-2, -1)
        ) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn_scores = attn_scores + mask.unsqueeze(1).float()
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = torch.nn.functional.dropout(attn_probs, p=self.attn_drop, training=self.training)
        
        context = torch.matmul(attn_probs, v_proj.float())
        context = context.transpose(1, 2).reshape(B, Nq, self.embed_dim)
        context = self.out_proj(context.to(q_proj.dtype))
        context = self.proj_drop(context)
        return context


class DTEMMergeOnly(DTEMBlock):
    """
    DTEM merge-only helper (no attention/MLP parameters).
    Reuses DTEMBlock's selection + merge logic without creating unused params.
    """

    def __init__(self):
        nn.Module.__init__(self)


# =====================================================================
# Transformer Block (for LoT and LaM)
# =====================================================================

class TransformerBlock(GradientCheckpointingLayer):
    """Standard Transformer block with RMSNorm and RoPE."""

    def __init__(self, config: MergeNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Use fla's fused RMSNorm if GPU available (faster), else PyTorch native
        self.attn_norm = get_rmsnorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=None,  # Full attention for LoT/LaM
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )

        self.mlp_norm = get_rmsnorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=None,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if use_cache:
            outputs += (past_key_values,)
        return outputs


# =====================================================================
# 1. Shared Local Transformer (LoT)
# =====================================================================

class SharedLocalTransformer(nn.Module):
    """
    Shared Local Transformer: extracts local context from byte sequences.
    Uses causal attention with RoPE.
    """

    def __init__(self, config: MergeNetConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        
        # Byte embedding
        self.embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=self.padding_idx
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_local_layers)
        ])
        
        # Final norm
        self.norm = get_rmsnorm(config.hidden_size, eps=config.norm_eps)
        
        self.gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, output_attentions=False):
        """
        Args:
            input_ids: (B, L) - byte token ids
            attention_mask: (B, L) - optional attention mask
            past_key_values: Optional[Cache] - cached key/value states
            use_cache: bool - whether to return key/value states for caching
            output_attentions: bool - whether to return attention weights
        Returns:
            hidden_states: (B, L, d) - local context representations
            next_cache: Optional[Cache] - updated cache (if use_cache=True)
        """
        # Convert legacy cache format if needed
        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)
        
        # Embedding
        hidden_states = self.embeddings(input_ids)
        
        next_cache = None
        
        # Forward through transformer blocks
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            
            if use_cache:
                # Cache is at index 2 if output_attentions, else index 1
                next_cache = layer_outputs[2 if output_attentions else 1]
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        if use_cache:
            return hidden_states, next_cache
        else:
            return hidden_states


# =====================================================================
# 2. Local Encoder (LoE) with DTEM 
# =====================================================================

class LocalEncoderNLP(nn.Module):
    """
    Local Encoder: differentiable soft tokenization using DTEM.
    Reuses timm/dtem.py implementation with timm-style Blocks.
    """

    def __init__(self, config: MergeNetConfig):
        super().__init__()
        self.config = config
        if config.local_depth <= 0:
            raise ValueError("local_depth must be >= 1")

        self.local_depth = config.local_depth

        # Local Transformer Blocks for full-length token encoding.
        local_window = config.dtem_window_size if config.dtem_window_size is not None else 16
        # Use pure Python to support meta-device model initialization in flame.
        if self.local_depth == 1:
            dpr = [float(config.drop_path_rate)]
        else:
            denom = float(max(self.local_depth - 1, 1))
            dpr = [float(config.drop_path_rate) * i / denom for i in range(self.local_depth)]
        self.blocks = nn.ModuleList([
            LocalBlock(
                dim=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=config.intermediate_size / config.hidden_size,
                qkv_bias=config.qkv_bias,
                qk_norm=config.qk_norm,
                proj_drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm,
                local_window=local_window,
            )
            for i in range(self.local_depth)
        ])

        # DTEM metric head (one per local layer), same as CV merge-only logic.
        self.metric_dim = self._resolve_metric_dim(config.hidden_size, config.num_heads, config.dtem_feat_dim)
        self.metric_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, self.metric_dim) for _ in range(self.local_depth)
        ])

        # Reuse DTEM merge logic only.
        self.merge_block = DTEMMergeOnly()
        window_size = config.dtem_window_size if config.dtem_window_size is not None else 0
        self._tome_info = {
            "r": None,
            "size": None,
            "source_map": None,
            "source_matrix": None,
            "total_merge": None,
            "trace_source": True,
            "prop_attn": True,
            "class_token": False,  # NLP has no cls token.
            "distill_token": False,
            "source_tracking_mode": "matrix",
            "window_size": window_size,
            "t": config.dtem_t,
            "use_softkmax": config.use_softkmax,
            "tau1": 1.0,
            "tau2": 30.0,
            "feat_dim": self.metric_dim,
            "local_depth": self.local_depth,
            "swa_size": config.dtem_window_size,
        }
        self.merge_block._tome_info = self._tome_info
        self.default_r = 0

    @staticmethod
    def _resolve_metric_dim(embed_dim: int, num_heads: int, dtem_feat_dim):
        if dtem_feat_dim is not None:
            return dtem_feat_dim
        head_dim = embed_dim // num_heads
        return head_dim if embed_dim < 1024 else 2 * head_dim

    def _aggregate_with_source_matrix(self, x, size, source_matrix):
        if source_matrix is None:
            return x
        center = self._tome_info["source_matrix_center"]
        width = self._tome_info["source_matrix_width"]
        _, N, _ = x.shape
        device = x.device

        # Memory-saving path: avoid materializing (B, N, width, C).
        summed = torch.zeros_like(x)
        base_positions = torch.arange(N, device=device)
        for offset in range(width):
            pos = base_positions + (offset - center)
            valid = (pos >= 0) & (pos < N)
            pos_clamped = pos.clamp(0, N - 1)
            gathered = x[:, pos_clamped, :]  # (B, N, C)
            weight = source_matrix[:, :, offset]
            if not valid.all():
                weight = weight * valid.to(x.dtype)
            summed = summed + gathered * weight.unsqueeze(-1)

        if size is not None:
            denom = size.squeeze(-1).clamp(min=1e-6).unsqueeze(-1)
        else:
            denom = source_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return summed / denom

    def forward(self, hidden_states, phase='phase2'):
        """
        Args:
            hidden_states: (B, L, d) - from LoT
            phase: 'phase1' (reconstruction) or 'phase2' (prediction)
        Returns:
            merged_states: (B, N, d) - compressed latent words
            size: (B, N, 1) - token sizes
            source_matrix: (B, N, width) - source tracking
            info: dict - additional information
        """
        del phase  # kept for backward-compatible signature
        B, L, _ = hidden_states.shape

        x_layers = []
        x = hidden_states
        for local_blk in self.blocks:
            x = local_blk(x)
            x_layers.append(x)
        if not x_layers:
            raise RuntimeError("LocalEncoderNLP requires at least one local block.")

        x_embed = x_layers[-1]
        x_merge = x_embed
        n = x_merge.shape[1]

        total_merge = int(L * (1 - 1 / self.config.lambda_local))
        self.default_r = total_merge // max(self.local_depth, 1)
        r_list = parse_r(self.local_depth, self.default_r, total_merge)

        self._tome_info["r"] = r_list
        self._tome_info["total_merge"] = total_merge
        self._tome_info["size"] = torch.ones_like(x_merge[..., 0:1])
        self._tome_info["token_counts_local"] = []

        size = self._tome_info["size"]
        source_matrix = None
        for i, layer_x in enumerate(x_layers):
            x_metric = self._aggregate_with_source_matrix(layer_x, size, source_matrix)
            metric = self.metric_layers[i](x_metric.detach())
            r = r_list[i] if i < len(r_list) else 0

            x_merge, size, n, _, source_matrix = self.merge_block._merge_train(
                x_merge, size, r, n, {"metric": metric}, source_matrix
            )
            self._tome_info["size"] = size
            self._tome_info["token_counts_local"].append(x_merge.shape[1])

            # Guard against NaN/Inf from DTEM merging (debug stability).
            if not torch.isfinite(x_merge).all():
                warnings.warn(
                    "Non-finite values detected in LocalEncoderNLP output; "
                    "applying nan_to_num fallback.",
                    stacklevel=2,
                )
                x_merge = torch.nan_to_num(x_merge, nan=0.0, posinf=1e4, neginf=-1e4)
            if size is not None and not torch.isfinite(size).all():
                size = torch.nan_to_num(size, nan=0.0, posinf=1e4, neginf=-1e4)

        self._tome_info["source_matrix"] = source_matrix
        self._tome_info["x_embed"] = x_embed
        return x_merge, size, source_matrix, self._tome_info


# =====================================================================
# 3. Latent Model (LaM) - Pure GPT Transformer
# =====================================================================

class LatentModel(nn.Module):
    """
    Latent Model: standard GPT-style transformer operating on compressed
    latent word sequences. No ToMe merging.
    """

    def __init__(self, config: MergeNetConfig):
        super().__init__()
        self.config = config
        
        # Pure GPT transformer layers (no ToMe)
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_latent_layers)
        ])
        
        # Final norm
        self.norm = get_rmsnorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

    def forward(self, latent_words, size=None, attention_mask=None, past_key_values=None, use_cache=False, output_attentions=False):
        """
        Args:
            latent_words: (B, N, d) - compressed tokens from LoE
            size: (B, N, 1) - optional token sizes (unused, kept for interface compatibility)
            attention_mask: (B, N) - optional attention mask
            past_key_values: Optional[Cache] - cached key/value states
            use_cache: bool - whether to return key/value states for caching
            output_attentions: bool - whether to return attention weights
        Returns:
            outputs: (B, N, d) - predicted latent representations
            next_cache: Optional[Cache] - updated cache (if use_cache=True)
        """
        # Convert legacy cache format if needed
        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)
        
        hidden_states = latent_words
        
        next_cache = None
        
        # Forward through transformer blocks
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            
            if use_cache:
                # Cache is at index 2 if output_attentions, else index 1
                next_cache = layer_outputs[2 if output_attentions else 1]
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        if use_cache:
            return hidden_states, next_cache
        else:
            return hidden_states


# =====================================================================
# 4. Local Decoder (LoD) with Band Mask
# =====================================================================

class LocalDecoder(nn.Module):
    """
    Local Decoder: decodes latent words back to byte level using
    cross-attention with band mask and grid bias.
    """

    def __init__(self, config: MergeNetConfig):
        super().__init__()
        self.config = config
        
        # Cross-attention from queries (byte-level) to latent words
        self.cross_attention = MyCrossAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            bias=config.qkv_bias,
            attn_drop=config.attn_drop_rate,
            proj_drop=config.drop_rate,
        )
        
        self.lambda_local = config.lambda_local
        self.W_infer = config.W_infer
        self.gamma = config.grid_bias_gamma

    def build_decoder_bias(self, query_states, latent_words, byte_positions=None, latent_positions=None):
        B, L, _ = query_states.shape
        N = latent_words.shape[1]
        device = query_states.device
        dtype = query_states.dtype

        if byte_positions is None:
            byte_positions = torch.arange(L, device=device)
        if latent_positions is None:
            latent_positions = torch.arange(N, device=device)

        t_indices = byte_positions.to(device=device, dtype=dtype).view(1, L, 1)
        j_indices = latent_positions.to(device=device, dtype=dtype).view(1, 1, N)

        # Grid bias: -gamma * |t/lambda - j|
        grid_bias = -self.gamma * torch.abs(t_indices / self.lambda_local - j_indices)

        # Causal sliding band in latent index space.
        latent_cursor = torch.floor(t_indices / self.lambda_local)
        cond_causal = j_indices <= latent_cursor
        cond_window = j_indices >= (latent_cursor - self.W_infer)
        invalid = torch.full_like(grid_bias, -1e10)
        mask_val = torch.where(cond_causal & cond_window, torch.zeros_like(grid_bias), invalid)
        return grid_bias + mask_val

    def forward(self, query_states, latent_words, byte_positions=None, latent_positions=None):
        """
        Args:
            query_states: (B, L, d) - original H from LoT
            latent_words: (B, N, d) - O from LaM
            byte_positions: (L,) or None - absolute positions of bytes (for generation)
            latent_positions: (N,) or None - absolute positions of latent words (for generation)
        Returns:
            decoded_states: (B, L, d) - byte-level representations
        """
        combined_mask = self.build_decoder_bias(
            query_states,
            latent_words,
            byte_positions=byte_positions,
            latent_positions=latent_positions,
        )
        # Cross-attention: Q from bytes, KV from latent words
        decoded_states = self.cross_attention(query_states, latent_words, mask=combined_mask)
        
        return decoded_states


# =====================================================================
# 5. Main Model
# =====================================================================

class MergeNetPreTrainedModel(PreTrainedModel):
    config_class = MergeNetConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['TransformerBlock', 'DTEMBlock']

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class MergeNetModel(MergeNetPreTrainedModel):
    """
    MergeNet base model outputting raw hidden states.
    """

    def __init__(self, config: MergeNetConfig):
        super().__init__(config)
        self.config = config
        
        # Module 1: Shared Local Transformer (LoT)
        self.shared_local_transformer = SharedLocalTransformer(config)
        
        # Module 2: Local Encoder (LoE) with DTEM
        self.local_encoder = LocalEncoderNLP(config)
        
        # Module 3: Latent Model (LaM) - Pure GPT
        self.latent_model = LatentModel(config)
        
        # Module 4: Local Decoder (LoD)
        self.local_decoder = LocalDecoder(config)
        
        # Cross-attention for Perceiver-style refinement
        self.encode_cross_attention = MyCrossAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            bias=config.qkv_bias,
            attn_drop=config.attn_drop_rate,
            proj_drop=config.drop_rate,
        )
        
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            input_ids: (B, L) - byte token ids
            attention_mask: (B, L) - optional attention mask
        Returns:
            BaseModelOutputWithPast
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Stage 1: Local Context Extraction (LoT)
        H = self.shared_local_transformer(input_ids, attention_mask)  # (B, L, d)
        H_original = H.clone()  # Save for LoD query
        
        # Stage 2: Soft Token Merging (LoE)
        Z_merged, size, source_matrix, info_local = self.local_encoder(H, phase=self.config.phase)
        x_embed = info_local.get("x_embed", H_original)
        # Z_merged: (B, N_merged, d), N_merged ≈ L - total_merge
        
        # Stage 3: TopK Selection + Perceiver Cross-Attention
        Z, size_z = self._select_and_refine_tokens(
            Z_merged, size, source_matrix, x_embed, info_local
        )
        # Z: (B, N, d), N = target latent words count
        
        # Stage 4: Latent Sequence Modeling (LaM)
        O = self.latent_model(Z, size=size_z, attention_mask=None)  # (B, N, d)
        
        # Stage 5: Decode back to Byte Level (LoD)
        H_decoded = self.local_decoder(H_original, O)  # (B, L, d)
        
        return BaseModelOutputWithPast(
            last_hidden_state=H_decoded,
            hidden_states=None,
            attentions=None,
        )
    
    def _select_and_refine_tokens(
        self,
        merged_tokens: torch.Tensor,  # (B, N_merged, d)
        size: torch.Tensor,  # (B, N_merged, 1)
        source_matrix: Optional[torch.Tensor],  # (B, N_merged, width)
        original_hidden: torch.Tensor,  # (B, L, d)
        info_local: dict,
    ):
        """
        Select top-k tokens by size, sort by center of mass, and refine via Perceiver.
        
        Args:
            merged_tokens: Tokens after DTEM merging
            size: Token sizes (weights)
            source_matrix: Source tracking matrix
            original_hidden: Original H from LoT
            info_local: Info dict from LoE
        Returns:
            refined_tokens: (B, N, d)
            refined_size: (B, N, 1)
        """
        B, N_merged, d = merged_tokens.shape
        L = original_hidden.shape[1]
        device = merged_tokens.device
        
        # Compute center of mass for each token based on source_matrix
        if source_matrix is not None:
            with torch.no_grad():
                center = info_local["source_matrix_center"]
                width = info_local["source_matrix_width"]
                
                # Token positions
                i_positions = torch.arange(N_merged, device=device).unsqueeze(0).expand(B, -1)  # (B, N_merged)
                
                # Relative offsets
                offset_relative = torch.arange(width, device=device, dtype=torch.float32) - center  # (width,)
                
                # Weighted offset: sum(source_matrix * offset)
                weighted_offset = (source_matrix * offset_relative.view(1, 1, -1)).sum(dim=-1)  # (B, N_merged)
                
                # Center of mass = position + weighted_offset / size
                token_center_of_mass = i_positions.float() + weighted_offset / size[..., 0].detach().clamp(min=1e-6)
        else:
            # Fallback: use position indices
            token_center_of_mass = torch.arange(N_merged, device=device).unsqueeze(0).expand(B, -1).float()
        
        # Target latent count should align with LoE merge result
        k = L - info_local.get("total_merge", 0)
        k = max(1, min(k, N_merged))  # Ensure k is valid
        
        # TopK selection by token size (strength)
        token_strength = size[..., 0]  # (B, N_merged)
        with torch.no_grad():
            topk_vals, topk_indices = torch.topk(token_strength.detach(), k, dim=1, largest=True, sorted=False)
            
            # Gather selected tokens' center of mass
            topk_com = torch.gather(token_center_of_mass, 1, topk_indices)  # (B, k)
            
            # Sort by center of mass to maintain spatial order
            sorted_order = torch.argsort(topk_com, dim=1)
            sorted_topk_indices = torch.gather(topk_indices, 1, sorted_order)  # (B, k)
        
        # Gather selected tokens
        topk_tokens = torch.gather(
            merged_tokens, 1, 
            sorted_topk_indices.unsqueeze(-1).expand(-1, -1, d)
        )  # (B, k, d)
        topk_size = torch.gather(size, 1, sorted_topk_indices.unsqueeze(-1))  # (B, k, 1)
        
        # Construct attention bias using log(source_matrix)
        bias = self._construct_perceiver_bias(
            sorted_topk_indices, source_matrix, info_local, L, k, device, merged_tokens.dtype
        )
        
        # Perceiver Cross-Attention: refine selected tokens using full context
        # Q: selected tokens, KV: original H
        refined_tokens = self.encode_cross_attention(topk_tokens, original_hidden, mask=bias)
        
        # Residual connection
        refined_tokens = refined_tokens + topk_tokens
        
        return refined_tokens, topk_size
    
    def _construct_perceiver_bias(
        self,
        selected_indices: torch.Tensor,  # (B, k)
        source_matrix: Optional[torch.Tensor],  # (B, N_merged, width)
        info_local: dict,
        L: int,
        k: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Construct log(source_matrix) as attention bias for Perceiver cross-attention.
        
        Returns:
            bias: (B, k, L) - attention bias
        """
        B = selected_indices.shape[0]
        
        if source_matrix is None:
            # No bias
            return None
        
        with torch.no_grad():
            center = info_local["source_matrix_center"]
            width = info_local["source_matrix_width"]
            
            # Initialize bias with large negative values (cannot attend)
            bias = torch.full((B, k, L), -1e10, device=device, dtype=dtype)
            
            # Extract source_matrix for selected tokens
            # source_matrix: (B, N_merged, width)
            # selected_indices: (B, k)
            source_for_selected = torch.gather(
                source_matrix, 1,
                selected_indices.unsqueeze(-1).expand(-1, -1, width)
            )  # (B, k, width)
            
            # Compute positions: j = selected_indices[i] + (offset - center)
            offset_range = torch.arange(width, device=device).view(1, 1, -1)  # (1, 1, width)
            j_positions = selected_indices.unsqueeze(-1) + (offset_range - center)  # (B, k, width)
            
            # Valid mask: positions within [0, L)
            valid_mask = (j_positions >= 0) & (j_positions < L)  # (B, k, width)
            
            # Take log of source values
            log_source = torch.where(
                source_for_selected > 1e-10,
                torch.log(source_for_selected.clamp(min=1e-10)),
                torch.full_like(source_for_selected, -1e10)
            )
            
            # Apply valid mask
            log_source_masked = torch.where(valid_mask, log_source, torch.full_like(log_source, -1e10))
            
            # Clamp j_positions for safe indexing
            j_positions_safe = torch.where(valid_mask, j_positions, torch.zeros_like(j_positions))
            
            # Scatter log_source into bias
            bias.scatter_(2, j_positions_safe, log_source_masked)
        
        return bias


class MergeNetForCausalLM(MergeNetPreTrainedModel):
    """
    MergeNet for causal language modeling (byte-level).
    """

    def __init__(self, config: MergeNetConfig):
        super().__init__(config)
        self.config = config
        
        # Base model
        self.model = MergeNetModel(config)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Loss criterion
        self.criterion = None
        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.shared_local_transformer.embeddings
    
    def set_input_embeddings(self, value):
        self.model.shared_local_transformer.embeddings = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = None,
        **kwargs,
    ):
        """
        Args:
            input_ids: (B, L) - byte token ids
            attention_mask: (B, L) - optional attention mask
            labels: (B, L) - optional labels for language modeling
        Returns:
            CausalLMOutputWithPast
        """
        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state  # (B, L, d)
        
        # Compute logits
        logits = self.lm_head(hidden_states)  # (B, L, vocab_size)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Initialize criterion if needed
            if self.criterion is None:
                self.criterion = nn.CrossEntropyLoss()
            
            # Shift for autoregressive prediction
            # Predict token t+1 from context 0:t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten and compute loss
            loss = self.criterion(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        use_sliding_window: bool = True,
        **kwargs,
    ):
        """
        Generate text using sliding window queue mechanism.
        
        Args:
            input_ids: (B, L_prompt) - prompt token ids
            max_length: maximum generation length
            temperature: sampling temperature
            top_p: nucleus sampling threshold
            top_k: top-k sampling threshold
            use_sliding_window: whether to use efficient sliding window (True) or full recompute (False)
        Returns:
            generated_ids: (B, max_length)
        """
        if use_sliding_window:
            return self._generate_with_sliding_window(
                input_ids, max_length, temperature, top_p, top_k, **kwargs
            )
        else:
            return self._generate_full_recompute(
                input_ids, max_length, temperature, top_p, top_k, **kwargs
            )
    
    def _generate_full_recompute(
        self,
        input_ids: torch.LongTensor,
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        **kwargs,
    ):
        """Fallback generation: recompute everything each step (simpler but slower)."""
        B, L_prompt = input_ids.shape
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids.clone()
        
        for step in range(max_length - L_prompt):
            # Forward pass through full model
            outputs = self.model(generated, attention_mask=None)
            hidden_states = outputs.last_hidden_state
            
            # Get logits for last position
            logits = self.lm_head(hidden_states[:, -1, :])  # (B, vocab_size)
            
            # Sample next token
            next_token = self._sample_next_token(logits, temperature, top_p, top_k)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if (next_token == self.config.eos_token_id).all():
                break
        
        return generated
    
    def _generate_with_sliding_window(
        self,
        input_ids: torch.LongTensor,
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        **kwargs,
    ):
        """
        Efficient generation using sliding window queue.
        
        Key insight: LoE is just a tokenizer (bytes -> latent words), only used once!
        After initialization, LaM autoregressively generates new latent words.
        
        Architecture flow:
        1. [Init once] Prompt -> LoT -> LoE -> LaM -> O_queue
        2. [Every step] New byte -> LoT -> H, then H (Q) + O_queue (KV) -> LocalDecoder -> prediction
        3. [Every λ steps] LaM autoregressively generates next latent word, update O_queue (sliding window)
        
        Complexity per step:
        - LoT: O(L * d^2) where L=window_size (local attention)
        - LocalDecoder: O(1 * N * d^2) where N=L/λ (cross-attention, query length=1)
        - LaM (every λ steps): O(N * d^2) amortized to O(N * d^2 / λ) per step
        Total: O(L * d^2 + N * d^2) ≈ O(L * d^2) per step
        """
        B, L_prompt = input_ids.shape
        device = input_ids.device
        
        lambda_local = self.config.lambda_local
        # Latent queue size: W_infer + 1 (for sufficient context in LocalDecoder)
        N_queue = self.config.W_infer + 1
        
        # === Initialization: Process FULL prompt (no truncation) ===
        # Step 1: LoT with KV cache - get byte-level representations for entire prompt
        # Since LoT uses global causal attention, we need to maintain full history
        # Use KV cache to avoid O(T²) recomputation
        H_full, past_key_values_lot = self.model.shared_local_transformer(
            input_ids, attention_mask=None, past_key_values=None, use_cache=True
        )  # (B, L_prompt, d)
        
        # Step 2: LoE - soft token merging with DTEM
        Z_merged, size_merged, source_matrix, info_local = self.model.local_encoder(H_full, phase=self.config.phase)
        x_embed = info_local.get("x_embed", H_full)
        # Z_merged: (B, N_merged, d), N_merged ≈ L - total_merge
        
        # Step 3: TopK Selection + Perceiver Cross-Attention
        Z_full, _ = self.model._select_and_refine_tokens(
            Z_merged, size_merged, source_matrix, x_embed, info_local
        )
        # Z_full: (B, M, d), M = L_prompt / λ (target latent count)

        # Step 4: Build latent outputs O_full and LaM cache from prompt latent words.
        O_full, past_key_values_lam = self.model.latent_model(
            Z_full, size=None, attention_mask=None, past_key_values=None, use_cache=True
        )
        M_total = Z_full.shape[1]

        # Sliding latent queue stores O only (consistent with training-time decoder KV).
        if M_total >= N_queue:
            O_queue = O_full[:, -N_queue:, :].clone()
            latent_positions = torch.arange(M_total - N_queue, M_total, device=device, dtype=torch.long)
        else:
            pad_len = N_queue - M_total
            O_queue = torch.cat([
                torch.zeros(B, pad_len, O_full.shape[2], device=device, dtype=O_full.dtype),
                O_full,
            ], dim=1)
            latent_positions = torch.cat([
                torch.full((pad_len,), -10**9, device=device, dtype=torch.long),
                torch.arange(M_total, device=device, dtype=torch.long),
            ], dim=0)

        # Maintain full generated byte sequence for API compatibility.
        byte_sequence = input_ids.clone()
        current_pos = L_prompt

        # Autoregressive latent generation state.
        next_latent_input = O_full[:, -1:, :].clone() if M_total > 0 else None
        next_latent_idx = M_total

        for step in range(max_length - L_prompt):
            # Decode next byte using current byte hidden state as query.
            if step == 0:
                H_current = H_full[:, -1:, :]
            else:
                last_byte = byte_sequence[:, -1:].clone()
                H_current, past_key_values_lot = self.model.shared_local_transformer(
                    last_byte,
                    attention_mask=None,
                    past_key_values=past_key_values_lot,
                    use_cache=True,
                )

            byte_pos = torch.tensor([current_pos - 1], device=device, dtype=torch.long)
            H_decoded = self.model.local_decoder(
                H_current,
                O_queue,
                byte_positions=byte_pos,
                latent_positions=latent_positions,
            )
            logits = self.lm_head(H_decoded[:, 0, :])
            next_token = self._sample_next_token(logits, temperature, top_p, top_k)

            if (next_token == self.config.eos_token_id).all():
                break

            byte_sequence = torch.cat([byte_sequence, next_token], dim=1)
            current_pos += 1

            # Ensure latent queue covers the decoder receptive field for next query.
            # Next query uses byte index (current_pos - 1).
            required_latent_idx = int(math.floor((current_pos - 1) / lambda_local))
            while next_latent_input is not None and next_latent_idx <= required_latent_idx:
                O_new, past_key_values_lam = self.model.latent_model(
                    next_latent_input,
                    size=None,
                    attention_mask=None,
                    past_key_values=past_key_values_lam,
                    use_cache=True,
                )

                O_queue = torch.cat([O_queue[:, 1:, :], O_new], dim=1)
                latent_positions = torch.cat([
                    latent_positions[1:],
                    torch.tensor([next_latent_idx], device=device, dtype=torch.long),
                ], dim=0)

                next_latent_input = O_new
                next_latent_idx += 1

        return byte_sequence
    
    def _sample_next_token(self, logits, temperature, top_p, top_k):
        """Sample next token from logits with temperature, top-p, top-k filtering."""
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample next token
        # For safety with untrained models, check for nan/inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # Use greedy decoding as fallback
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
        else:
            probs = torch.softmax(logits, dim=-1)
            # Additional safety: clamp probs to avoid numerical issues
            probs = torch.clamp(probs, min=1e-8, max=1.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
            try:
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            except RuntimeError:
                # Fallback to greedy if multinomial fails
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
        
        return next_token

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

