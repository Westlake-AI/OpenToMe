import math
from typing import Dict

import torch

from ..base import KVCompressionPolicy
from .common import gather_tokens, group_queries
from .selectors.scissorhands import select_scissorhands_tokens


class ScissorhandsPolicy(KVCompressionPolicy):
    """Persistence-of-importance eviction with protected sink/recent tokens."""

    def __init__(self, config):
        super().__init__(config)
        self._importance: Dict[int, torch.Tensor] = {}

    def _scores(self, query_states, key_states):
        queries = group_queries(query_states, key_states.shape[1])
        logits = torch.einsum("bkgqd,bksd->bkgqs", queries, key_states)
        return torch.softmax(logits.float() / math.sqrt(key_states.shape[-1]), dim=-1).mean(dim=2).sum(dim=-2)

    def _select(self, key_states, value_states, importance, layer_idx):
        capacity = self.capacity_for_layer(layer_idx)
        if key_states.shape[-2] <= capacity:
            self._importance[layer_idx] = importance
            return key_states, value_states
        generator = torch.Generator(device=key_states.device).manual_seed(
            self.config.random_seed + layer_idx
        )
        indices = select_scissorhands_tokens(
            importance,
            token_budget=capacity,
            recent_size=self.config.window_size,
            sink_size=self.config.sink_size,
            selection=self.config.scissorhands_selection,
            generator=generator,
            random_temperature=self.config.random_temperature,
        )
        self._importance[layer_idx] = importance.gather(dim=-1, index=indices)
        return gather_tokens(key_states, indices), gather_tokens(value_states, indices)

    def compress_prefill(self, key_states, value_states, query_states, layer_idx):
        return self._select(
            key_states, value_states, self._scores(query_states, key_states), layer_idx
        )

    def compress_decode(self, key_states, value_states, query_states, layer_idx):
        new_length = query_states.shape[-2]
        current = self._scores(query_states, key_states)
        previous = self._importance.get(layer_idx)
        if previous is None or previous.shape[-1] != key_states.shape[-2] - new_length:
            previous = current.new_zeros(current.shape[:-1] + (key_states.shape[-2] - new_length,))
        previous = torch.cat(
            (previous, current.new_zeros(current.shape[:-1] + (new_length,))), dim=-1
        )
        importance = previous * self.config.scissorhands_decay + current
        return self._select(key_states, value_states, importance, layer_idx)


__all__ = ["ScissorhandsPolicy"]
