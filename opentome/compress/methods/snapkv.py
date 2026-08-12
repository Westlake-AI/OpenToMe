from ..base import KVCompressionPolicy
from .common import snap_compress


class SnapKVPolicy(KVCompressionPolicy):
    """Observation-window attention pooling and per-KV-head top-k selection."""

    def compress_prefill(self, key_states, value_states, query_states, layer_idx):
        return snap_compress(
            key_states,
            value_states,
            query_states,
            self.capacity_for_layer(layer_idx),
            self.config.window_size,
            self.config.kernel_size,
            self.config.pooling,
        )


__all__ = ["SnapKVPolicy"]
