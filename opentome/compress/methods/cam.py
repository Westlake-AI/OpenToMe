import math

import torch

from ..base import KVCompressionPolicy
from .common import gather_tokens, group_queries


class CAMPolicy(KVCompressionPolicy):
    """Attention-informed cache merging followed by prompt top-k retention."""

    def compress_prefill(self, key_states, value_states, query_states, layer_idx):
        capacity = self.capacity_for_layer(layer_idx)
        sequence_length = key_states.shape[-2]
        if sequence_length <= capacity:
            return key_states, value_states
        window = min(self.config.window_size, capacity - 1)
        queries = group_queries(query_states, key_states.shape[1])
        logits = torch.einsum("bkgqd,bksd->bkgqs", queries, key_states)
        logits = logits / math.sqrt(key_states.shape[-1])
        causal = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=logits.device),
            diagonal=1,
        )
        probabilities = torch.softmax(
            logits.masked_fill(causal, torch.finfo(logits.dtype).min).float(), dim=-1
        ).mean(dim=2)
        prefix_scores = probabilities[..., :-window].sum(dim=-2)
        indices = prefix_scores.topk(capacity - window, dim=-1).indices.sort(dim=-1).values

        # Merge each evicted prefix value into its closest retained key.
        selected_keys = gather_tokens(key_states[..., :-window, :], indices)
        selected_values = gather_tokens(value_states[..., :-window, :], indices).clone()
        keep_mask = torch.zeros_like(prefix_scores, dtype=torch.bool)
        keep_mask.scatter_(dim=-1, index=indices, value=True)
        for batch in range(key_states.shape[0]):
            for head in range(key_states.shape[1]):
                evicted = torch.nonzero(~keep_mask[batch, head], as_tuple=False).flatten()
                if not evicted.numel():
                    continue
                similarity = torch.matmul(
                    key_states[batch, head, evicted], selected_keys[batch, head].transpose(0, 1)
                )
                destinations = similarity.argmax(dim=-1)
                weights = prefix_scores[batch, head, evicted].to(selected_values.dtype)
                weighted = value_states[batch, head, evicted] * weights.unsqueeze(-1)
                selected_values[batch, head].index_add_(0, destinations, weighted)
        return (
            torch.cat((selected_keys, key_states[..., -window:, :]), dim=-2),
            torch.cat((selected_values, value_states[..., -window:, :]), dim=-2),
        )


__all__ = ["CAMPolicy"]
