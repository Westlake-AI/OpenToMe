from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from .base import KVCompressionConfig
from .methods import build_policy


class CompressedDynamicCache(DynamicCache):
    """A Transformers DynamicCache whose stored KV tensors are policy-managed."""

    def __init__(self, config: KVCompressionConfig):
        super().__init__()
        self.compression_config = config
        self.policy = build_policy(config)
        self.logical_lengths: Dict[int, int] = {}
        self._opentome_seen_tokens = 0

    @property
    def seen_tokens(self) -> int:
        """Logical token count kept for compatibility with older Transformers."""
        return self._opentome_seen_tokens

    @seen_tokens.setter
    def seen_tokens(self, value: int) -> None:
        self._opentome_seen_tokens = value

    def _get_layer_pair(self, layer_idx: int):
        if hasattr(self, "layers"):
            if layer_idx >= len(self.layers):
                return None
            layer = self.layers[layer_idx]
            if not getattr(layer, "is_initialized", False):
                return None
            return layer.keys, layer.values

        if layer_idx >= len(self.key_cache):
            return None
        keys = self.key_cache[layer_idx]
        if not isinstance(keys, torch.Tensor):
            return None
        return keys, self.value_cache[layer_idx]

    def _set_layer_pair(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if hasattr(self, "layers"):
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())
            layer = self.layers[layer_idx]
            if not getattr(layer, "is_initialized", False):
                layer.lazy_initialization(keys)
            layer.keys = keys
            layer.values = values
            return

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append([])
            self.value_cache.append([])
        self.key_cache[layer_idx] = keys
        self.value_cache[layer_idx] = values

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_kwargs = cache_kwargs or {}
        query_states = cache_kwargs.get("query_states")
        if query_states is None:
            raise ValueError(
                "CompressedDynamicCache requires an OpenToMe model adapter; "
                "patch the model before inference"
            )

        new_tokens = key_states.shape[-2]
        if layer_idx == 0:
            self._opentome_seen_tokens += new_tokens
        self.logical_lengths[layer_idx] = self.logical_lengths.get(layer_idx, 0) + new_tokens

        cached_pair = self._get_layer_pair(layer_idx)
        if cached_pair is None:
            stored_keys, stored_values = self.policy.compress_prefill(
                key_states, value_states, query_states, layer_idx
            )
            self._set_layer_pair(layer_idx, stored_keys, stored_values)
            return key_states, value_states

        cached_keys, cached_values = cached_pair
        keys = torch.cat((cached_keys, key_states), dim=-2)
        values = torch.cat((cached_values, value_states), dim=-2)
        stored_keys, stored_values, attention_keys, attention_values = self.policy.update_decode(
            keys, values, query_states, layer_idx
        )
        self._set_layer_pair(layer_idx, stored_keys, stored_values)
        return attention_keys, attention_values

    def get_logical_length(self, layer_idx: int = 0) -> int:
        return self.logical_lengths.get(layer_idx, 0)

    def cache_bytes(self) -> int:
        pairs = (self._get_layer_pair(i) for i in range(len(self)))
        tensors = [tensor for pair in pairs if pair is not None for tensor in pair]
        return sum(t.numel() * t.element_size() for t in tensors)

    def layer_lengths(self):
        pairs = (self._get_layer_pair(i) for i in range(len(self)))
        return [pair[0].shape[-2] for pair in pairs if pair is not None]
