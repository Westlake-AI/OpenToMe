from .snapkv import SnapKVPolicy


class PyramidKVPolicy(SnapKVPolicy):
    """SnapKV selection with a linearly decreasing layer-wise cache budget."""

    def capacity_for_layer(self, layer_idx: int) -> int:
        num_layers = self.config.num_hidden_layers
        if num_layers == 1:
            multiplier = 1.0
        else:
            progress = layer_idx / (num_layers - 1)
            multiplier = 1.0 + self.config.pyramid_beta * (1.0 - 2.0 * progress)
        return max(
            self.config.window_size + 1,
            int(round(self.config.max_capacity_prompt * multiplier)),
        )


__all__ = ["PyramidKVPolicy"]
