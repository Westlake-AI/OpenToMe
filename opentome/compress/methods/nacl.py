import math

import torch

from ..base import KVCompressionPolicy
from .common import gather_tokens, group_queries
from .selectors.nacl import select_nacl_proxy_indices, select_nacl_tokens


class NACLPolicy(KVCompressionPolicy):
    """One-shot proxy-token scoring with optional randomized retention."""

    def compress_prefill(self, key_states, value_states, query_states, layer_idx):
        capacity = self.capacity_for_layer(layer_idx)
        sequence_length = key_states.shape[-2]
        if sequence_length <= capacity:
            return key_states, value_states

        queries = group_queries(query_states, key_states.shape[1])
        logits = torch.einsum("bkgqd,bksd->bkgqs", queries, key_states)
        logits = logits / math.sqrt(key_states.shape[-1])
        causal = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=logits.device),
            diagonal=1,
        )
        logits.masked_fill_(causal, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(logits.float(), dim=-1).mean(dim=2)
        proxy = select_nacl_proxy_indices(
            sequence_length,
            proxy_size=self.config.nacl_proxy_size,
            mode=self.config.nacl_proxy_mode,
            sink_size=self.config.sink_size,
            device=key_states.device,
        )
        recent = torch.arange(
            max(0, sequence_length - self.config.window_size),
            sequence_length,
            device=key_states.device,
        )
        generator = torch.Generator(device=key_states.device).manual_seed(
            self.config.random_seed + layer_idx
        )
        indices = select_nacl_tokens(
            probabilities,
            token_budget=capacity,
            proxy_indices=proxy,
            protected_indices=recent,
            random_budget=self.config.nacl_random_budget,
            generator=generator,
            random_temperature=self.config.random_temperature,
        )
        return gather_tokens(key_states, indices), gather_tokens(value_states, indices)


__all__ = ["NACLPolicy"]
