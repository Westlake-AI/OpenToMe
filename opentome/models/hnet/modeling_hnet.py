"""HNet – Hierarchical byte-level language model
=================================================

Re-implementation of H-Net (https://github.com/goombalab/hnet) as a
HuggingFace-compatible ``PreTrainedModel`` so that it can be trained with
the *flame* trainer and evaluated with the standard OpenToMe / FLA
toolchain.

Architecture overview (two-stage example)
------------------------------------------

.. code-block:: text

    bytes ─▸ Embedding ─▸ HNet (stage-0) ─▸ lm_head ─▸ logits
                           │
                           ├─ Encoder (Isotropic: Mamba2/MHA stack)
                           ├─ RoutingModule (cosine-sim boundary pred)
                           ├─ ChunkLayer   (select boundary tokens)
                           ├─ HNet (stage-1, innermost)
                           │    └─ Isotropic (MHA/Mamba2 stack)
                           ├─ DeChunkLayer (EMA de-aggregator via Mamba2 kernel)
                           ├─ residual (STE-gated)
                           └─ Decoder (Isotropic: Mamba2/MHA stack)

Key design choices
-------------------
* The recursive ``HNet`` module is kept as a plain ``nn.Module`` and wrapped
  by ``HNetModel`` / ``HNetForCausalLM`` which inherit from
  ``PreTrainedModel``.
* Supports both *padded* (mask-based, ``B×L×D``) and *packed* (cu_seqlens)
  modes – exactly like the original hnet code.
* Load-balancing loss for the routing module is computed inside
  ``HNetForCausalLM.forward()`` and added to the main CE loss.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from opentome.models.hnet.configuration_hnet import HNetConfig
from opentome.models.utils import FLAGenerationMixin

try:
    from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
    from fla.modules.l2warp import l2_warp

    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False
    RMSNorm = None

try:
    from flash_attn import (
        flash_attn_qkvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
    )
    from flash_attn.ops.triton.layer_norm import RMSNorm as TritonRMSNorm

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False
    TritonRMSNorm = None

try:
    from flash_attn.ops.activations import swiglu as _fused_swiglu

    _HAS_FUSED_SWIGLU = True
except ImportError:
    _HAS_FUSED_SWIGLU = False

try:
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    _HAS_MAMBA = True
except ImportError:
    _HAS_MAMBA = False
    Mamba2 = None

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    try:
        from fla.models.modeling_layers import GradientCheckpointingLayer
    except ImportError:
        GradientCheckpointingLayer = nn.Module

logger = logging.get_logger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _get_norm_cls(fuse: bool = True, eps: float = 1e-5, **factory_kwargs):
    """Return a partial-applied norm constructor."""
    if fuse and TritonRMSNorm is not None:
        return partial(TritonRMSNorm, eps=eps, **factory_kwargs)
    elif fuse and RMSNorm is not None:
        return partial(RMSNorm, eps=eps)
    else:
        return partial(nn.RMSNorm, eps=eps)


def _get_seq_idx(cu_seqlens: torch.Tensor, device=None) -> torch.Tensor:
    """Compute per-token sequence index from cu_seqlens."""
    seq_idx = torch.zeros(cu_seqlens[-1], dtype=torch.long, device=device)
    seq_idx[cu_seqlens[:-1]] = 1
    seq_idx = (torch.cumsum(seq_idx, dim=0) - 1).unsqueeze(0).int()
    return seq_idx


def _get_stage_cfg(cfg: dict, stage_idx: int) -> dict:
    """Extract per-stage config values from potentially list-valued dicts."""
    return {k: v[stage_idx] if isinstance(v, (list, tuple)) else v for k, v in cfg.items()}


# ---------------------------------------------------------------------------
# STE – Straight-Through Estimator
# ---------------------------------------------------------------------------


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _ste_func(x):
    return _STE.apply(x)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------


class HNetSwiGLU(nn.Module):
    """SwiGLU feed-forward network (matches the original hnet ``SwiGLU``)."""

    def __init__(
        self,
        d_model: int,
        d_intermediate: int | None = None,
        bias: bool = False,
        multiple_of: int = 128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        d_intermediate = d_intermediate if d_intermediate is not None else int(8 * d_model / 3)
        d_intermediate = (d_intermediate + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(d_model, 2 * d_intermediate, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(d_intermediate, d_model, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        if _HAS_FUSED_SWIGLU:
            y = _fused_swiglu(gate, y)
        else:
            y = F.silu(gate) * y
        y = self.fc2(y)
        return y


# ---------------------------------------------------------------------------
# Rotary Embedding (copied from original hnet for exact compatibility)
# ---------------------------------------------------------------------------


class HNetRotaryEmbedding(nn.Module):
    """Rotary position embedding (RoFormer / RoPE)."""

    def __init__(self, dim: int, base: float = 10000.0, interleaved: bool = False, device=None):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale = None  # no xPos

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seqlen: int, device=None, dtype=None):
        if seqlen > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            if self.inv_freq.dtype != torch.float32:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
            else:
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(self, qkv, seqlen_offset=0, cu_seqlens=None, max_seqlen=None, **kwargs):
        """Apply rotary embedding *inplace* to Q and K in the qkv tensor.

        qkv: (..., 3, H, D) or (total, 3, H, D) when packed.
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
            try:
                from flash_attn.ops.triton.rotary import apply_rotary

                qk = qkv[:, :2].reshape(qkv.shape[0], -1, qkv.shape[-1])
                apply_rotary(
                    qk,
                    self._cos_cached,
                    self._sin_cached,
                    seqlen_offsets=0,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    interleaved=self.interleaved,
                    inplace=True,
                )
            except ImportError:
                # Fallback: apply rotary manually (slow path)
                pass
            return qkv

        seqlen = qkv.shape[1]
        self._update_cos_sin_cache(seqlen + (seqlen_offset if isinstance(seqlen_offset, int) else 0), device=qkv.device, dtype=qkv.dtype)

        try:
            from hnet.modules.rotary import apply_rotary_emb_qkv_

            return apply_rotary_emb_qkv_(
                qkv,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
            )
        except ImportError:
            try:
                from flash_attn.ops.triton.rotary import apply_rotary

                qk = qkv[:, :, :2].reshape(qkv.shape[0], qkv.shape[1], -1, qkv.shape[-1])
                apply_rotary(
                    qk,
                    self._cos_cached,
                    self._sin_cached,
                    seqlen_offsets=seqlen_offset,
                    interleaved=self.interleaved,
                    inplace=True,
                )
            except ImportError:
                pass
            return qkv


# ---------------------------------------------------------------------------
# Causal Multi-Head Attention
# ---------------------------------------------------------------------------


class HNetCausalMHA(nn.Module):
    """Multi-head causal self-attention with flash-attn backend."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        qkv_proj_bias: bool = False,
        out_proj_bias: bool = False,
        window_size: int = -1,
        softmax_scale=None,
        layer_idx=None,
        rotary_emb_dim: int = 0,
        rotary_emb_base: float = 10000.0,
        rotary_emb_interleaved: bool = False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale
        self.rotary_emb_dim = rotary_emb_dim
        self.window_size = window_size
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        qkv_dim = self.head_dim * 3 * num_heads

        if rotary_emb_dim > 0:
            self.rotary_emb = HNetRotaryEmbedding(rotary_emb_dim, base=rotary_emb_base, interleaved=rotary_emb_interleaved, device=device)

        self.Wqkv = nn.Linear(d_model, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(batch_size, max_seqlen, 2, self.num_heads, self.head_dim, dtype=dtype, device=device)

    def forward(self, x, cu_seqlens=None, max_seqlen=None, inference_params=None, **kwargs):
        if cu_seqlens is not None:
            assert max_seqlen is not None

        seqlen_offset = 0 if inference_params is None else (
            inference_params.lengths_per_sample if inference_params.lengths_per_sample is not None else inference_params.seqlen_offset
        )

        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)

        if self.rotary_emb_dim > 0:
            qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        if inference_params is None:
            # Training / prefill
            if cu_seqlens is not None and _HAS_FLASH_ATTN:
                context = flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens.int(), int(max_seqlen),
                    softmax_scale=self.softmax_scale, causal=True,
                    window_size=(self.window_size, -1),
                )
            elif _HAS_FLASH_ATTN:
                context = flash_attn_qkvpacked_func(
                    qkv, softmax_scale=self.softmax_scale, causal=True,
                    window_size=(self.window_size, -1),
                )
            else:
                # Fallback: naive attention
                q, k, v = qkv.unbind(dim=-3)
                scale = self.softmax_scale or (self.head_dim ** -0.5)
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                # Causal mask
                L = q.shape[-2]
                causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=q.device), diagonal=1)
                attn = attn + causal_mask
                attn = F.softmax(attn, dim=-1)
                context = torch.matmul(attn, v)
        else:
            # Step/decode path - not needed for training
            raise NotImplementedError("Inference step through HNetCausalMHA not implemented in this port")

        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out

    def step(self, x, inference_params):
        return self.forward(x, inference_params=inference_params)


# ---------------------------------------------------------------------------
# Mamba2 Wrapper
# ---------------------------------------------------------------------------


class Mamba2Wrapper(nn.Module):
    """Wrapper around mamba_ssm.Mamba2 that provides a step() interface."""

    def __init__(self, d_model: int, layer_idx: int = None, device=None, dtype=None, **ssm_kwargs):
        super().__init__()
        assert _HAS_MAMBA, "mamba_ssm is required for Mamba2 layers"
        self.inner = Mamba2(d_model=d_model, layer_idx=layer_idx, device=device, dtype=dtype, **ssm_kwargs)
        self.layer_idx = layer_idx

    def forward(self, hidden_states, inference_params=None, **kwargs):
        return self.inner(hidden_states, inference_params=inference_params, **kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.inner.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def step(self, hidden_states, inference_params):
        conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
        result, conv_state, ssm_state = self.inner.step(hidden_states, conv_state, ssm_state)
        inference_params.key_value_memory_dict[self.layer_idx][0].copy_(conv_state)
        inference_params.key_value_memory_dict[self.layer_idx][1].copy_(ssm_state)
        return result


# ---------------------------------------------------------------------------
# Block (pre-norm mixer + optional MLP)
# ---------------------------------------------------------------------------


class HNetBlock(nn.Module):
    """Pre-norm block: RMSNorm → Mixer → [RMSNorm → MLP]."""

    def __init__(self, d_model, mixer_cls, mlp_cls=None, norm_cls=None, residual_in_fp32=True):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        if norm_cls is None:
            norm_cls = partial(nn.RMSNorm, eps=1e-5)
        self.norm1 = norm_cls(d_model)
        self.mixer = mixer_cls(d_model)
        if mlp_cls is not None and mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(d_model)
            self.mlp = mlp_cls(d_model)
        else:
            self.mlp = None

    def forward(self, hidden_states, residual=None, inference_params=None, mixer_kwargs=None):
        # Handle both Triton RMSNorm (prenorm=True returns (out, residual))
        # and plain nn.RMSNorm (single return)
        norm1_out = self.norm1(hidden_states, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32) \
            if hasattr(self.norm1, 'prenorm') or (TritonRMSNorm is not None and isinstance(self.norm1, TritonRMSNorm)) \
            else None

        if norm1_out is not None:
            hidden_states, residual = norm1_out
        else:
            if residual is not None:
                hidden_states = hidden_states + residual
            residual = hidden_states.float() if self.residual_in_fp32 else hidden_states
            hidden_states = self.norm1(hidden_states)

        if mixer_kwargs is None:
            mixer_kwargs = {}
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            norm2_out = self.norm2(hidden_states, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32) \
                if hasattr(self.norm2, 'prenorm') or (TritonRMSNorm is not None and isinstance(self.norm2, TritonRMSNorm)) \
                else None
            if norm2_out is not None:
                hidden_states, residual = norm2_out
            else:
                hidden_states = hidden_states + residual
                residual = hidden_states.float() if self.residual_in_fp32 else hidden_states
                hidden_states = self.norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def step(self, hidden_states, inference_params, residual=None):
        norm1_out = self.norm1(hidden_states, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32) \
            if hasattr(self.norm1, 'prenorm') or (TritonRMSNorm is not None and isinstance(self.norm1, TritonRMSNorm)) \
            else None
        if norm1_out is not None:
            hidden_states, residual = norm1_out
        else:
            if residual is not None:
                hidden_states = hidden_states + residual
            residual = hidden_states.float() if self.residual_in_fp32 else hidden_states
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.mixer.step(hidden_states, inference_params)

        if self.mlp is not None:
            norm2_out = self.norm2(hidden_states, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32) \
                if hasattr(self.norm2, 'prenorm') or (TritonRMSNorm is not None and isinstance(self.norm2, TritonRMSNorm)) \
                else None
            if norm2_out is not None:
                hidden_states, residual = norm2_out
            else:
                hidden_states = hidden_states + residual
                residual = hidden_states.float() if self.residual_in_fp32 else hidden_states
                hidden_states = self.norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


def create_hnet_block(
    arch: str,
    d_model: int,
    d_intermediate: int = 0,
    ssm_cfg: dict | None = None,
    attn_cfg: dict | None = None,
    norm_epsilon: float = 1e-5,
    layer_idx: int = None,
    residual_in_fp32: bool = True,
    device=None,
    dtype=None,
) -> HNetBlock:
    """Create a single HNetBlock based on architecture character.

    ``arch`` is one of:
        ``t`` – MHA only (no MLP)
        ``T`` – MHA + SwiGLU MLP
        ``m`` – Mamba2 only (no MLP)
        ``M`` – Mamba2 + SwiGLU MLP
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_cfg is None:
        attn_cfg = {}

    # Mixer
    if arch in ("t", "T"):
        mixer_cls = partial(HNetCausalMHA, **attn_cfg, **factory_kwargs, layer_idx=layer_idx)
    elif arch in ("m", "M"):
        mixer_cls = partial(Mamba2Wrapper, **ssm_cfg, **factory_kwargs, layer_idx=layer_idx)
    else:
        raise NotImplementedError(f"Unknown arch char: {arch}")

    # MLP
    if arch in ("T", "M"):
        mlp_cls = partial(HNetSwiGLU, d_intermediate=d_intermediate, **factory_kwargs)
    elif arch in ("t", "m"):
        mlp_cls = nn.Identity
    else:
        raise NotImplementedError

    # Norm
    norm_cls = _get_norm_cls(fuse=True, eps=norm_epsilon, **factory_kwargs)

    return HNetBlock(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls, residual_in_fp32=residual_in_fp32)


# ---------------------------------------------------------------------------
# Isotropic stack (flat sequence of blocks)
# ---------------------------------------------------------------------------


@dataclass
class IsotropicInferenceParams:
    max_seqlen: int = 0
    max_batch_size: int = 0
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None


class HNetIsotropic(nn.Module):
    """A flat stack of HNetBlocks, parsed from an arch layout string."""

    def __init__(self, config: HNetConfig, pos_idx: int, stage_idx: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]
        ssm_cfg = config.get_stage_ssm_cfg(stage_idx)
        attn_cfg = config.get_stage_attn_cfg(stage_idx)

        # Navigate to the correct position in the arch_layout
        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]
        arch_layout = arch_layout[pos_idx]

        # Parse layout string like "m4", "T1m4", "T26"
        layout_parse = re.findall(r"([mMtT])(\d+)", arch_layout)

        layers = []
        layer_idx = 0
        self.arch_full = []
        self.height = 0

        for arch, n_layer in layout_parse:
            n = int(n_layer)
            layers += [
                create_hnet_block(
                    arch,
                    self.d_model,
                    d_intermediate=config.d_intermediate[stage_idx],
                    ssm_cfg=ssm_cfg,
                    attn_cfg=attn_cfg,
                    layer_idx=layer_idx + i,
                    **factory_kwargs,
                )
                for i in range(n)
            ]
            self.height += n if arch.islower() else 2 * n
            self.arch_full.extend([arch] * n)
            layer_idx += n

        self.layers = nn.ModuleList(layers)

        norm_cls = _get_norm_cls(fuse=True, eps=1e-5, **factory_kwargs)
        self.rmsnorm = norm_cls(self.d_model)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        kv_mem = {}
        for i, layer in enumerate(self.layers):
            kv_mem[i] = layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
        return IsotropicInferenceParams(key_value_memory_dict=kv_mem, max_seqlen=max_seqlen, max_batch_size=batch_size)

    def forward(self, hidden_states, cu_seqlens=None, max_seqlen=None, mask=None, inference_params=None, **mixer_kwargs):
        assert (mask is not None) or (cu_seqlens is not None and max_seqlen is not None)

        import copy
        attn_mixer_kwargs = copy.deepcopy(mixer_kwargs)
        ssm_mixer_kwargs = copy.deepcopy(mixer_kwargs)

        if mask is not None:
            packed = False
        else:
            attn_mixer_kwargs.update({"cu_seqlens": cu_seqlens.int(), "max_seqlen": max_seqlen})
            ssm_mixer_kwargs.update({"seq_idx": _get_seq_idx(cu_seqlens, device=hidden_states.device)})
            packed = True

        residual = None
        for layer, arch in zip(self.layers, self.arch_full):
            if arch in ("m", "M"):
                layer_mixer_kwargs = ssm_mixer_kwargs
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                    residual = None if residual is None else residual.unsqueeze(0)
            elif arch in ("t", "T"):
                layer_mixer_kwargs = attn_mixer_kwargs
                if hidden_states.dim() == 3 and packed:
                    hidden_states = hidden_states.squeeze(0)
                    residual = None if residual is None else residual.squeeze(0)
            else:
                raise NotImplementedError

            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params, mixer_kwargs=layer_mixer_kwargs)

        # Final norm
        if TritonRMSNorm is not None and isinstance(self.rmsnorm, TritonRMSNorm):
            hidden_states = self.rmsnorm(hidden_states, residual=residual, prenorm=False, residual_in_fp32=True)
        else:
            if residual is not None:
                hidden_states = hidden_states + residual
            hidden_states = self.rmsnorm(hidden_states)

        if hidden_states.dim() == 3 and packed:
            hidden_states = hidden_states.squeeze(0)

        if inference_params is not None:
            assert mask.shape[0] == 1
            inference_params.seqlen_offset += hidden_states.shape[1]

        return hidden_states

    def step(self, hidden_states, inference_params):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer.step(hidden_states, inference_params, residual=residual)

        if TritonRMSNorm is not None and isinstance(self.rmsnorm, TritonRMSNorm):
            hidden_states = self.rmsnorm(hidden_states, residual=residual, prenorm=False, residual_in_fp32=True)
        else:
            if residual is not None:
                hidden_states = hidden_states + residual
            hidden_states = self.rmsnorm(hidden_states)

        inference_params.seqlen_offset += 1
        return hidden_states


# ---------------------------------------------------------------------------
# Routing / Chunk / DeChunk modules
# ---------------------------------------------------------------------------


@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor


@dataclass
class RoutingModuleState:
    has_seen_tokens: torch.Tensor
    last_hidden_state: torch.Tensor


@dataclass
class DeChunkState:
    last_value: torch.Tensor


class HNetRoutingModule(nn.Module):
    """Cosine-similarity based boundary predictor."""

    def __init__(self, d_model: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        # Initialize to identity
        with torch.no_grad():
            nn.init.eye_(self.q_proj_layer.weight)
            nn.init.eye_(self.k_proj_layer.weight)
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
        )

    def forward(self, hidden_states, cu_seqlens=None, mask=None, inference_params=None):
        assert (mask is not None) or (cu_seqlens is not None)

        if cu_seqlens is not None:
            hidden_states = hidden_states.unsqueeze(0)

        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
        )
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        if cu_seqlens is not None:
            boundary_prob = boundary_prob.squeeze(0)
            boundary_prob[cu_seqlens[:-1]] = PAD_PROB

        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)
        selected_idx = torch.argmax(boundary_prob, dim=-1)
        boundary_mask = selected_idx == 1

        if mask is not None:
            boundary_mask = boundary_mask & mask

        if inference_params is not None:
            has_mask = mask.any(dim=-1)
            inference_params.has_seen_tokens.copy_(has_mask | inference_params.has_seen_tokens)
            last_mask = torch.clamp(mask.sum(dim=-1) - 1, min=0)
            inference_params.last_hidden_state.copy_(
                torch.where(
                    has_mask,
                    hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), last_mask],
                    inference_params.last_hidden_state,
                )
            )

        selected_probs = boundary_prob.gather(dim=-1, index=selected_idx.unsqueeze(-1))

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )

    def step(self, hidden_states, inference_params):
        hidden_states_sq = hidden_states.squeeze(1)
        cos_sim = torch.einsum(
            "b d, b d -> b",
            F.normalize(self.q_proj_layer(inference_params.last_hidden_state), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states_sq), dim=-1),
        )
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        inference_params.last_hidden_state.copy_(hidden_states_sq)
        boundary_prob = torch.where(inference_params.has_seen_tokens, boundary_prob, torch.ones_like(boundary_prob))
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)
        inference_params.has_seen_tokens.copy_(torch.ones_like(inference_params.has_seen_tokens))

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_prob[..., 1] > 0.5,
            selected_probs=boundary_prob.max(dim=-1).values.unsqueeze(-1),
        )


class HNetChunkLayer(nn.Module):
    """Select tokens at boundary positions to create chunks."""

    def forward(self, hidden_states, boundary_mask, cu_seqlens=None, mask=None):
        assert (mask is not None) or (cu_seqlens is not None)

        if cu_seqlens is not None:
            next_hidden_states = hidden_states[boundary_mask]
            next_cu_seqlens = F.pad(boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0))
            next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())
            next_mask = None
        else:
            next_cu_seqlens = None
            num_tokens = boundary_mask.sum(dim=-1)
            next_max_seqlen = int(num_tokens.max())
            device = hidden_states.device
            L = hidden_states.shape[1]
            token_idx = torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
            seq_sorted_indices = torch.argsort(token_idx, dim=1)
            next_hidden_states = torch.gather(
                hidden_states, dim=1,
                index=seq_sorted_indices[:, :next_max_seqlen, None].expand(-1, -1, hidden_states.shape[-1]),
            )
            next_mask = torch.arange(next_max_seqlen, device=device)[None, :] < num_tokens[:, None]
            next_max_seqlen = None

        return next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask

    def step(self, hidden_states, boundary_mask):
        return hidden_states[boundary_mask]


class HNetDeChunkLayer(nn.Module):
    """EMA-based de-aggregator using the Mamba2 scan kernel."""

    def __init__(self, d_model: int, dtype=torch.bfloat16, block_size: int = 256, headdim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype
        self.block_size = block_size
        self.headdim = headdim
        assert d_model % headdim == 0
        self.nheads = d_model // headdim

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return DeChunkState(last_value=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype))

    def forward(self, hidden_states, boundary_mask, boundary_prob, cu_seqlens=None, inference_params=None, mask=None):
        assert _HAS_MAMBA, "mamba_ssm is required for DeChunkLayer (mamba_chunk_scan_combined)"

        p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - 1e-4)

        if cu_seqlens is not None:
            p = p[boundary_mask].unsqueeze(0)
            seq_idx = _get_seq_idx(cu_seqlens, device=hidden_states.device)
        else:
            B, L = boundary_mask.shape
            seq_idx = None
            token_idx = torch.arange(L, device=hidden_states.device)[None, :] + (~boundary_mask).long() * L
            seq_sorted_indices = torch.argsort(token_idx, dim=1)
            p = torch.gather(p, dim=1, index=seq_sorted_indices[:, :hidden_states.shape[1]])

        original_dtype = hidden_states.dtype
        dt = torch.log(1 / (1 - p)).to(self.dtype)
        x = (hidden_states / dt[..., None]).to(self.dtype)
        A = -torch.ones((self.nheads,), device=hidden_states.device, dtype=torch.float32)
        b = p.to(self.dtype)
        c = torch.ones_like(b)

        out = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            repeat(dt, "b l -> b l h", h=self.nheads),
            A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1"),
            chunk_size=self.block_size,
            seq_idx=seq_idx,
        )
        out = rearrange(out, "b l h p -> b l (h p)")

        if cu_seqlens is not None:
            out = out.squeeze(0)
            plug_back_idx = boundary_mask.cumsum(dim=0) - 1
            out = torch.gather(out, dim=0, index=plug_back_idx.unsqueeze(-1).expand(-1, self.d_model))
        else:
            plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1
            out = torch.gather(out, dim=1, index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model))

        if inference_params is not None:
            inference_params.last_value.copy_(out[:, -1])

        return out.to(original_dtype)

    def step(self, hidden_states, boundary_mask, boundary_prob, inference_params):
        B = boundary_mask.shape[0]
        D = hidden_states.shape[-1] if hidden_states.numel() > 0 else self.d_model

        p = torch.zeros(B, device=boundary_mask.device, dtype=inference_params.last_value.dtype)
        p[boundary_mask] = boundary_prob[boundary_mask, -1].clamp(min=1e-4, max=1 - 1e-4)

        current = torch.zeros(B, D, device=boundary_mask.device, dtype=inference_params.last_value.dtype)
        if hidden_states.numel() > 0:
            current[boundary_mask] = hidden_states.squeeze(1)

        result = p * current + (1 - p) * inference_params.last_value
        inference_params.last_value.copy_(result)
        return result.unsqueeze(1)


# ---------------------------------------------------------------------------
# HNet – recursive hierarchical module
# ---------------------------------------------------------------------------


@dataclass
class HNetState:
    encoder_state: Optional[IsotropicInferenceParams] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[Any] = None  # HNetState or IsotropicInferenceParams
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[IsotropicInferenceParams] = None


class HNet(nn.Module):
    """Recursive hierarchical module."""

    def __init__(self, config: HNetConfig, stage_idx: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]

        assert isinstance(arch_layout, list), f"Wrong arch_layout: {arch_layout}"
        if len(arch_layout) == 3:
            sub_model_names = ["encoder", "main_network", "decoder"]
            self.is_innermost = False
        elif len(arch_layout) == 1:
            sub_model_names = ["main_network"]
            self.is_innermost = True
        else:
            raise NotImplementedError

        for _name, _layout in zip(sub_model_names, arch_layout):
            if self.is_innermost or _name in ("encoder", "decoder"):
                SubModel = HNetIsotropic
                _stage_idx = stage_idx
                _pos_idx = None
                if _name == "encoder":
                    _pos_idx = 0
                elif self.is_innermost:
                    _pos_idx = 0
                elif _name == "decoder":
                    _pos_idx = 2
                _pos_idx_dict = {"pos_idx": _pos_idx}
            else:
                SubModel = HNet
                _stage_idx = stage_idx + 1
                _pos_idx_dict = {}

            _sub_model = SubModel(config=config, stage_idx=_stage_idx, **_pos_idx_dict, **factory_kwargs)
            self.add_module(_name, _sub_model)

        if not self.is_innermost:
            self.routing_module = HNetRoutingModule(self.d_model, **factory_kwargs)
            self.chunk_layer = HNetChunkLayer()
            self.dechunk_layer = HNetDeChunkLayer(
                self.d_model,
                dtype=dtype if dtype is not None else torch.bfloat16,
                block_size=config.dechunk_block_size,
                headdim=config.dechunk_headdim,
            )
            # Residual projection in fp32
            self.residual_proj = nn.Linear(self.d_model, self.d_model, device=device, dtype=torch.float32)
            nn.init.zeros_(self.residual_proj.weight)
            self.residual_proj.weight._no_reinit = True
            self.residual_func = lambda out, residual, p: out * _ste_func(p) + residual

        # Dimension padding for inner stages
        if stage_idx > 0 and self.d_model - config.d_model[stage_idx - 1] > 0:
            self.pad_dimension = nn.Parameter(
                torch.zeros(self.d_model - config.d_model[stage_idx - 1], **factory_kwargs)
            )
        else:
            self.pad_dimension = None

    def _init_weights(self, initializer_range: float = 0.02, parent_residuals: int = 0):
        n_residuals = parent_residuals
        if self.is_innermost:
            n_residuals += self.main_network.height
            for name, m in self.main_network.named_modules():
                if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
        else:
            n_residuals += self.encoder.height + self.decoder.height
            for part in (self.encoder, self.decoder):
                for name, m in part.named_modules():
                    if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                        if "out_proj" in name or "fc2" in name:
                            nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                        else:
                            nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
            self.main_network._init_weights(initializer_range, n_residuals)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        if self.is_innermost:
            return HNetState(
                main_network_state=self.main_network.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            )
        else:
            device = self.residual_proj.weight.device
            return HNetState(
                encoder_state=self.encoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype),
                routing_module_state=self.routing_module.allocate_inference_cache(batch_size, max_seqlen, device, dtype=dtype),
                main_network_state=self.main_network.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype),
                dechunk_state=self.dechunk_layer.allocate_inference_cache(batch_size, max_seqlen, device, dtype=dtype),
                decoder_state=self.decoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype),
            )

    def forward(self, hidden_states, cu_seqlens=None, max_seqlen=None, mask=None, inference_params=None, **mixer_kwargs):
        assert mask is not None or (cu_seqlens is not None and max_seqlen is not None)

        if inference_params is None:
            inference_params = HNetState(main_network_state=None)
        else:
            assert mask is not None

        D = hidden_states.shape[-1]
        EARLY_DIMS = hidden_states.shape[:-1]

        if self.pad_dimension is not None:
            hidden_states = torch.cat((hidden_states, self.pad_dimension.expand(EARLY_DIMS + (-1,))), dim=-1)

        if self.is_innermost:
            hidden_states = self.main_network(
                hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                mask=mask, inference_params=inference_params.main_network_state, **mixer_kwargs,
            )
            return hidden_states[..., :D], []

        # Encoder
        hidden_states = self.encoder(
            hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            mask=mask, inference_params=inference_params.encoder_state, **mixer_kwargs,
        )

        # Residual
        hidden_states_for_residual = hidden_states.to(dtype=self.residual_proj.weight.dtype)
        residual = self.residual_proj(hidden_states_for_residual)

        # Routing
        bpred_output = self.routing_module(
            hidden_states, cu_seqlens=cu_seqlens, mask=mask,
            inference_params=inference_params.routing_module_state,
        )

        # Chunk
        hidden_states, next_cu_seqlens, next_max_seqlen, next_mask = self.chunk_layer(
            hidden_states, bpred_output.boundary_mask, cu_seqlens, mask=mask,
        )

        # Main network (recursive)
        hidden_states, prev_boundary_predictions = self.main_network(
            hidden_states, cu_seqlens=next_cu_seqlens, max_seqlen=next_max_seqlen,
            mask=next_mask, inference_params=inference_params.main_network_state, **mixer_kwargs,
        )

        # DeChunk
        hidden_states = self.dechunk_layer(
            hidden_states, bpred_output.boundary_mask, bpred_output.boundary_prob,
            next_cu_seqlens, mask=mask, inference_params=inference_params.dechunk_state,
        )

        # Residual with STE
        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype), residual, bpred_output.selected_probs
        ).to(hidden_states.dtype)

        # Decoder
        hidden_states = self.decoder(
            hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            mask=mask, inference_params=inference_params.decoder_state, **mixer_kwargs,
        )

        return hidden_states[..., :D], [bpred_output, *prev_boundary_predictions]

    def step(self, hidden_states, inference_params):
        D = hidden_states.shape[-1]
        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (hidden_states, self.pad_dimension.expand(hidden_states.shape[:-1] + (-1,))), dim=-1
            )

        if self.is_innermost:
            hidden_states = self.main_network.step(hidden_states, inference_params.main_network_state)
            return hidden_states[..., :D], []

        hidden_states = self.encoder.step(hidden_states, inference_params.encoder_state)
        hidden_states_for_residual = hidden_states.to(dtype=self.residual_proj.weight.dtype)
        residual = self.residual_proj(hidden_states_for_residual)

        bpred_output = self.routing_module.step(hidden_states, inference_params.routing_module_state)
        hidden_states_inner = self.chunk_layer.step(hidden_states, bpred_output.boundary_mask)

        if hidden_states_inner.shape[0] > 0:
            hidden_states_inner, prev_boundary_predictions = self.main_network.step(
                hidden_states_inner, inference_params.main_network_state
            )
        else:
            prev_boundary_predictions = []

        hidden_states = self.dechunk_layer.step(
            hidden_states_inner, bpred_output.boundary_mask, bpred_output.boundary_prob,
            inference_params.dechunk_state,
        )

        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype), residual, bpred_output.selected_probs
        ).to(hidden_states.dtype)

        hidden_states = self.decoder.step(hidden_states, inference_params.decoder_state)
        return hidden_states[..., :D], [bpred_output, *prev_boundary_predictions]


# ---------------------------------------------------------------------------
# Load-balancing loss
# ---------------------------------------------------------------------------


def load_balancing_loss(router_output: RoutingModuleOutput, N: float) -> torch.Tensor:
    """Compute load-balancing loss encouraging a 1/N compression ratio."""
    tokenized_prob = router_output.boundary_prob[..., -1]
    boundary_mask = router_output.boundary_mask

    true_ratio = boundary_mask.float().mean()
    average_prob = tokenized_prob.float().mean()

    return ((1 - true_ratio) * (1 - average_prob) + true_ratio * average_prob * (N - 1)) * N / (N - 1)


# ---------------------------------------------------------------------------
# HuggingFace PreTrainedModel wrappers
# ---------------------------------------------------------------------------


class HNetPreTrainedModel(PreTrainedModel):
    config_class = HNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HNetIsotropic", "HNetBlock"]
    _supports_cache_class = False

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            if not getattr(module.weight, "_no_reinit", False):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0)


class HNetModel(HNetPreTrainedModel):
    """Bare HNet model: embedding → HNet backbone → (hidden_states, bpred_outputs)."""

    def __init__(self, config: HNetConfig):
        super().__init__(config)
        self.config = config
        d_embed = config.d_model[0]

        self.embeddings = nn.Embedding(config.vocab_size, d_embed)
        self.backbone = HNet(config=config, stage_idx=0)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        B, L, D = inputs_embeds.shape

        # Determine mode: packed (cu_seqlens) or padded (mask)
        if attention_mask is not None:
            # Use mask-based (padded) mode
            mask = attention_mask.bool()
            hidden_states = inputs_embeds
            _cu_seqlens = None
            _max_seqlen = None
        else:
            # Use packed mode
            mask = None
            hidden_states = inputs_embeds.flatten(0, 1)  # (B*L, D)
            if cu_seqlens is not None:
                # Use externally provided cu_seqlens (e.g. from varlen dataloader)
                _cu_seqlens = cu_seqlens.flatten().int()
                _max_seqlen = int((_cu_seqlens[1:] - _cu_seqlens[:-1]).max())
            else:
                # Absent cu_seqlens, assume uniform packing
                _cu_seqlens = torch.arange(B + 1, device=hidden_states.device) * L
                _max_seqlen = L

        hidden_states, bpred_outputs = self.backbone(
            hidden_states,
            cu_seqlens=_cu_seqlens,
            max_seqlen=_max_seqlen,
            mask=mask,
        )

        hidden_states = hidden_states.view(B, L, D)
        return hidden_states, bpred_outputs


class HNetForCausalLM(HNetPreTrainedModel, FLAGenerationMixin):
    """HNet with a language modeling head for causal LM training."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: HNetConfig):
        super().__init__(config)
        self.model = HNetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.d_model[0], config.vocab_size, bias=False)
        self.criterion = None

        if config.tie_embeddings:
            self.lm_head.weight = self.model.embeddings.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, bpred_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cu_seqlens=cu_seqlens,
        )

        logits = None
        if not getattr(self.config, "fuse_linear_cross_entropy", False) or labels is None:
            logits = self.lm_head(
                hidden_states if logits_to_keep is None or logits_to_keep == 0
                else hidden_states[:, -logits_to_keep:]
            )

        loss = None
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if getattr(self.config, "fuse_linear_cross_entropy", False) and _HAS_FLA:
                    criterion = FusedLinearCrossEntropyLoss(
                        use_l2warp=getattr(self.config, "use_l2warp", False)
                    )
                elif getattr(self.config, "fuse_cross_entropy", True) and _HAS_FLA:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion

            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)

            if getattr(self.config, "fuse_linear_cross_entropy", False) and _HAS_FLA:
                ce_loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                ce_loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                if getattr(self.config, "use_l2warp", False) and _HAS_FLA:
                    ce_loss = l2_warp(ce_loss, logits)

            loss = ce_loss

            # Add load-balancing loss for the routing modules
            lb_weight = getattr(self.config, "lb_loss_weight", 0.0)
            lb_N = getattr(self.config, "lb_loss_N", 4.0)
            if lb_weight > 0 and len(bpred_outputs) > 0:
                lb_loss = sum(load_balancing_loss(bp, lb_N) for bp in bpred_outputs) / len(bpred_outputs)
                loss = loss + lb_weight * lb_loss

        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


__all__ = [
    "HNetPreTrainedModel",
    "HNetModel",
    "HNetForCausalLM",
    "HNet",
    "HNetIsotropic",
    "HNetBlock",
    "HNetRoutingModule",
    "HNetChunkLayer",
    "HNetDeChunkLayer",
]
