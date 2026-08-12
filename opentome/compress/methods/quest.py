from ..base import KVCompressionPolicy
from .common import gather_tokens, group_queries
from .selectors.quest import select_quest_tokens


class QuestPolicy(KVCompressionPolicy):
    """Query-aware page retrieval while retaining the full physical KV cache."""

    def compress_prefill(self, key_states, value_states, query_states, layer_idx):
        return key_states, value_states

    def update_decode(self, key_states, value_states, query_states, layer_idx):
        capacity = self.capacity_for_layer(layer_idx)
        if key_states.shape[-2] <= capacity:
            return key_states, value_states, key_states, value_states
        kv_queries = group_queries(query_states, key_states.shape[1]).mean(dim=2)
        indices = select_quest_tokens(
            kv_queries,
            key_states,
            page_size=self.config.quest_page_size,
            token_budget=capacity,
            recent_size=self.config.window_size,
        )
        return (
            key_states,
            value_states,
            gather_tokens(key_states, indices),
            gather_tokens(value_states, indices),
        )


__all__ = ["QuestPolicy"]
