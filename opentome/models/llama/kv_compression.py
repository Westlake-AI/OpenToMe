import math
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:  # Transformers 4.37
    ALL_ATTENTION_FUNCTIONS = None

from opentome.compress import CompressedDynamicCache


def _project_qkv(self, hidden_states, num_heads, num_kv_heads):
    pretraining_tp = getattr(self.config, "pretraining_tp", 1)
    if pretraining_tp <= 1:
        return (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

    kv_slice = (num_kv_heads * self.head_dim) // pretraining_tp
    query_slices = self.q_proj.weight.split(
        (num_heads * self.head_dim) // pretraining_tp, dim=0
    )
    key_slices = self.k_proj.weight.split(kv_slice, dim=0)
    value_slices = self.v_proj.weight.split(kv_slice, dim=0)
    return (
        torch.cat([F.linear(hidden_states, weight) for weight in query_slices], dim=-1),
        torch.cat([F.linear(hidden_states, weight) for weight in key_slices], dim=-1),
        torch.cat([F.linear(hidden_states, weight) for weight in value_slices], dim=-1),
    )


def _compressed_llama_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    position_embeddings=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Llama attention compatible with Transformers 4.37 and 4.57+."""
    batch_size, query_length, _ = hidden_states.size()
    num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
    num_kv_heads = getattr(
        self, "num_key_value_heads", self.config.num_key_value_heads
    )
    hidden_size = getattr(self, "hidden_size", self.config.hidden_size)
    cache = past_key_values if past_key_values is not None else past_key_value
    modern_api = position_embeddings is not None

    query_states, key_states, value_states = _project_qkv(
        self, hidden_states, num_heads, num_kv_heads
    )
    query_states = query_states.view(
        batch_size, query_length, num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        batch_size, query_length, num_kv_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        batch_size, query_length, num_kv_heads, self.head_dim
    ).transpose(1, 2)

    if modern_api:
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
    else:
        if isinstance(cache, CompressedDynamicCache):
            logical_past = cache.get_logical_length(self.layer_idx)
        elif cache is not None:
            logical_past = cache.get_usable_length(query_length, self.layer_idx)
        else:
            logical_past = 0
        rotary_length = logical_past + query_length
        if position_ids is not None:
            rotary_length = max(rotary_length, int(position_ids.max().item()) + 1)
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_length)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

    if cache is not None:
        key_states, value_states = cache.update(
            key_states,
            value_states,
            self.layer_idx,
            {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "query_states": query_states,
            },
        )

    attention_backend = getattr(self.config, "_attn_implementation", "eager")
    if modern_api and attention_backend != "eager":
        if ALL_ATTENTION_FUNCTIONS is None:
            raise RuntimeError(
                f"Attention backend {attention_backend!r} is unavailable in this Transformers version"
            )
        attention_output, attention_weights = ALL_ATTENTION_FUNCTIONS[attention_backend](
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=getattr(self, "scaling", self.head_dim ** -0.5),
            output_attentions=output_attentions,
            **kwargs,
        )
        attention_output = attention_output.reshape(
            batch_size, query_length, hidden_size
        ).contiguous()
        return self.o_proj(attention_output), attention_weights

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    kv_length = key_states.shape[-2]
    scaling = getattr(self, "scaling", self.head_dim ** -0.5)
    attention_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) * scaling
    if attention_mask is not None:
        mask = attention_mask[..., :kv_length] if modern_api else attention_mask[..., -kv_length:]
        attention_weights = attention_weights + mask
    attention_weights = nn.functional.softmax(
        attention_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attention_weights = nn.functional.dropout(
        attention_weights, p=self.attention_dropout, training=self.training
    )
    attention_output = torch.matmul(attention_weights, value_states)
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(
        batch_size, query_length, hidden_size
    )

    pretraining_tp = getattr(self.config, "pretraining_tp", 1)
    if pretraining_tp > 1:
        chunks = attention_output.split(hidden_size // pretraining_tp, dim=2)
        slices = self.o_proj.weight.split(hidden_size // pretraining_tp, dim=1)
        attention_output = sum(
            F.linear(chunks[i], slices[i]) for i in range(pretraining_tp)
        )
    else:
        attention_output = self.o_proj(attention_output)

    if not output_attentions:
        attention_weights = None
    if modern_api:
        return attention_output, attention_weights
    return attention_output, attention_weights, cache


def patch_llama_model(model):
    """Patch one Llama model instance for query-aware compressed cache use."""
    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise TypeError("Expected a Transformers Llama model with model.layers")
    for layer in layers:
        attention = layer.self_attn
        if not hasattr(attention, "_opentome_original_forward"):
            attention._opentome_original_forward = attention.forward
            attention.forward = MethodType(_compressed_llama_attention_forward, attention)
    return model


def unpatch_llama_model(model):
    """Restore attention methods previously replaced by patch_llama_model."""
    base_model = getattr(model, "model", model)
    for layer in base_model.layers:
        attention = layer.self_attn
        original = getattr(attention, "_opentome_original_forward", None)
        if original is not None:
            attention.forward = original
            del attention._opentome_original_forward
    return model


__all__ = ["patch_llama_model", "unpatch_llama_model"]
