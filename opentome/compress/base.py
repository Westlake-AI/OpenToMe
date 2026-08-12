from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class KVCompressionConfig:
    """Configuration shared by KV cache compression policies."""

    method: str = "snapkv"
    max_capacity_prompt: int = 2048
    window_size: int = 32
    kernel_size: int = 5
    pooling: str = "avgpool"
    sink_size: int = 4
    pyramid_beta: float = 0.5
    num_hidden_layers: int = 1
    quest_page_size: int = 16
    nacl_proxy_size: int = 32
    nacl_proxy_mode: str = "suffix"
    nacl_random_budget: int = 0
    scissorhands_decay: float = 1.0
    scissorhands_selection: str = "topk"
    random_seed: int = 42
    random_temperature: float = 1.0

    def __post_init__(self):
        method = self.method.lower()
        object.__setattr__(self, "method", method)
        if not method:
            raise ValueError("method must not be empty")
        if self.max_capacity_prompt <= 0:
            raise ValueError("max_capacity_prompt must be positive")
        if self.window_size <= 0 or self.window_size >= self.max_capacity_prompt:
            raise ValueError("window_size must be in [1, max_capacity_prompt)")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if self.pooling not in {"avgpool", "maxpool"}:
            raise ValueError("pooling must be 'avgpool' or 'maxpool'")
        if self.sink_size < 0:
            raise ValueError("sink_size must be non-negative")
        if method in {"streamingkv", "streamingllm"} and self.sink_size >= self.max_capacity_prompt:
            raise ValueError("sink_size must be smaller than max_capacity_prompt")
        if not 0.0 <= self.pyramid_beta < 1.0:
            raise ValueError("pyramid_beta must be in [0, 1)")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.quest_page_size <= 0:
            raise ValueError("quest_page_size must be positive")
        if self.nacl_proxy_size < 0 or self.nacl_random_budget < 0:
            raise ValueError("NACL proxy and random budgets must be non-negative")
        if self.nacl_proxy_mode not in {"suffix", "prefix", "edges"}:
            raise ValueError("nacl_proxy_mode must be suffix, prefix, or edges")
        if self.scissorhands_decay < 0:
            raise ValueError("scissorhands_decay must be non-negative")
        if self.scissorhands_selection not in {"topk", "prob"}:
            raise ValueError("scissorhands_selection must be topk or prob")
        if self.random_temperature < 0:
            raise ValueError("random_temperature must be non-negative")


TensorPair = Tuple[torch.Tensor, torch.Tensor]


class KVCompressionPolicy(ABC):
    """Interface implemented by every KV cache compression policy."""

    def __init__(self, config: KVCompressionConfig):
        self.config = config

    def capacity_for_layer(self, layer_idx: int) -> int:
        return self.config.max_capacity_prompt

    @abstractmethod
    def compress_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
    ) -> TensorPair:
        raise NotImplementedError

    def compress_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
    ) -> TensorPair:
        return key_states, value_states

    def update_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
    ):
        """Return storage K/V followed by K/V exposed to attention."""
        keys, values = self.compress_decode(
            key_states, value_states, query_states, layer_idx
        )
        return keys, values, keys, values
