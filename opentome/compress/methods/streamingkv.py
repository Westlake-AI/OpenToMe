import torch

from ..base import KVCompressionPolicy, TensorPair


class StreamingKVPolicy(KVCompressionPolicy):
    """Attention sinks plus a fixed-size rolling window."""

    def _compress(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> TensorPair:
        capacity = self.capacity_for_layer(layer_idx)
        if key_states.shape[-2] <= capacity:
            return key_states, value_states
        sink_size = min(self.config.sink_size, capacity - 1)
        recent_size = capacity - sink_size
        if not sink_size:
            return key_states[..., -recent_size:, :], value_states[..., -recent_size:, :]
        return (
            torch.cat((key_states[..., :sink_size, :], key_states[..., -recent_size:, :]), dim=-2),
            torch.cat((value_states[..., :sink_size, :], value_states[..., -recent_size:, :]), dim=-2),
        )

    def compress_prefill(self, key_states, value_states, query_states, layer_idx):
        return self._compress(key_states, value_states, layer_idx)

    def compress_decode(self, key_states, value_states, query_states, layer_idx):
        return self._compress(key_states, value_states, layer_idx)


__all__ = ["StreamingKVPolicy"]
