from ..base import KVCompressionPolicy
from .common import gather_tokens


class L2NormPolicy(KVCompressionPolicy):
    """Retain tokens with the smallest key L2 norms, matching KVCache-Factory."""

    def compress_prefill(self, key_states, value_states, query_states, layer_idx):
        capacity = self.capacity_for_layer(layer_idx)
        if key_states.shape[-2] <= capacity:
            return key_states, value_states
        indices = key_states.norm(p=2, dim=-1).argsort(dim=-1)[..., :capacity]
        indices = indices.sort(dim=-1).values
        return gather_tokens(key_states, indices), gather_tokens(value_states, indices)


__all__ = ["L2NormPolicy"]
