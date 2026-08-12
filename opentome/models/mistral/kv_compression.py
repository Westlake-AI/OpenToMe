import math
from types import MethodType

import torch
import torch.nn as nn
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:  # Transformers 4.37
    ALL_ATTENTION_FUNCTIONS = None

from opentome.compress import CompressedDynamicCache


def _compressed_mistral_attention_forward(
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
    """Mistral eager attention compatible with Transformers 4.37 and 4.57+."""
    batch_size, query_length, _ = hidden_states.size()
    num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
    num_kv_heads = getattr(
        self, "num_key_value_heads", self.config.num_key_value_heads
    )
    hidden_size = getattr(self, "hidden_size", self.config.hidden_size)
    cache = past_key_values if past_key_values is not None else past_key_value
    modern_api = position_embeddings is not None

    query_states = self.q_proj(hidden_states).view(
        batch_size, query_length, num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(
        batch_size, query_length, num_kv_heads, self.head_dim
    ).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(
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
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=getattr(self, "scaling", self.head_dim ** -0.5),
            output_attentions=output_attentions, **kwargs,
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
    attention_output = self.o_proj(attention_output)

    if not output_attentions:
        attention_weights = None
    if modern_api:
        return attention_output, attention_weights
    return attention_output, attention_weights, cache


def _compressed_mistral_model_forward(self, *args, **kwargs):
    cache = kwargs.get("past_key_values")
    attention_mask = kwargs.get("attention_mask")
    legacy_attention = hasattr(self.layers[0].self_attn, "rotary_emb")
    if (
        legacy_attention
        and isinstance(cache, CompressedDynamicCache)
        and attention_mask is not None
    ):
        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        if input_ids is None and args:
            input_ids = args[0]
        input_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        physical_length = cache.get_seq_length() + input_length
        if attention_mask.shape[-1] > physical_length:
            kwargs["attention_mask"] = attention_mask[..., -physical_length:]
    return self._opentome_original_forward(*args, **kwargs)


def patch_mistral_model(model):
    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise TypeError("Expected a Transformers Mistral model with model.layers")
    if not hasattr(base_model, "_opentome_original_forward"):
        base_model._opentome_original_forward = base_model.forward
        base_model.forward = MethodType(_compressed_mistral_model_forward, base_model)
    for layer in layers:
        attention = layer.self_attn
        if not hasattr(attention, "_opentome_original_forward"):
            attention._opentome_original_forward = attention.forward
            attention.forward = MethodType(_compressed_mistral_attention_forward, attention)
    return model


def unpatch_mistral_model(model):
    base_model = getattr(model, "model", model)
    original_model_forward = getattr(base_model, "_opentome_original_forward", None)
    if original_model_forward is not None:
        base_model.forward = original_model_forward
        del base_model._opentome_original_forward
    for layer in base_model.layers:
        attention = layer.self_attn
        original = getattr(attention, "_opentome_original_forward", None)
        if original is not None:
            attention.forward = original
            del attention._opentome_original_forward
    return model


__all__ = ["patch_mistral_model", "unpatch_mistral_model"]
