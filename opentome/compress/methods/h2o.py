import math
from typing import Dict

import torch

from ..base import KVCompressionPolicy
from .common import gather_tokens, group_queries, snap_compress


class H2OPolicy(KVCompressionPolicy):
    """Heavy-hitter eviction with a protected recent window."""

    def __init__(self, config):
        super().__init__(config)
        self._scores: Dict[int, torch.Tensor] = {}

    def compress_prefill(self, key_states, value_states, query_states, layer_idx):
        capacity = self.capacity_for_layer(layer_idx)
        if key_states.shape[-2] <= capacity:
            keys, values = key_states, value_states
        else:
            keys, values = snap_compress(
                key_states, value_states, query_states, capacity,
                self.config.window_size, 1, "avgpool",
            )
        self._scores[layer_idx] = keys.new_zeros(
            keys.shape[:2] + (keys.shape[-2],), dtype=torch.float32
        )
        return keys, values

    def compress_decode(self, key_states, value_states, query_states, layer_idx):
        queries = group_queries(query_states, key_states.shape[1])
        logits = torch.einsum("bkgqd,bksd->bkgqs", queries, key_states)
        probabilities = torch.softmax(logits.float() / math.sqrt(key_states.shape[-1]), dim=-1)
        step_scores = probabilities.sum(dim=(-2, -3))
        new_length = query_states.shape[-2]
        old_scores = self._scores.get(layer_idx)
        if old_scores is None or old_scores.shape[-1] != key_states.shape[-2] - new_length:
            old_scores = step_scores.new_zeros(
                step_scores.shape[:-1] + (key_states.shape[-2] - new_length,)
            )
        scores = torch.cat(
            (old_scores, step_scores.new_zeros(step_scores.shape[:-1] + (new_length,))), dim=-1
        ) + step_scores

        capacity = self.capacity_for_layer(layer_idx)
        if key_states.shape[-2] <= capacity:
            self._scores[layer_idx] = scores
            return key_states, value_states
        recent_size = min(self.config.window_size, capacity - 1)
        heavy_budget = capacity - recent_size
        heavy_scores = scores[..., :-recent_size]
        indices = heavy_scores.topk(heavy_budget, dim=-1).indices.sort(dim=-1).values
        kept_scores = heavy_scores.gather(dim=-1, index=indices)
        self._scores[layer_idx] = torch.cat((kept_scores, scores[..., -recent_size:]), dim=-1)
        return (
            torch.cat((gather_tokens(key_states[..., :-recent_size, :], indices), key_states[..., -recent_size:, :]), dim=-2),
            torch.cat((gather_tokens(value_states[..., :-recent_size, :], indices), value_states[..., -recent_size:, :]), dim=-2),
        )


__all__ = ["H2OPolicy"]
