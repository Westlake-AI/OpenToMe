import unittest

import torch
from transformers import MistralConfig, MistralForCausalLM

from opentome.compress import CompressedDynamicCache, KVCompressionConfig, POLICY_REGISTRY
from opentome.compress.methods.selectors.minicache import (
    compress_minicache_pair,
    minicache_slerp,
    restore_minicache_pair,
)
from opentome.compress.methods.selectors.nacl import (
    select_nacl_proxy_indices,
    select_nacl_tokens,
)
from opentome.compress.methods.selectors.quest import select_quest_tokens
from opentome.compress.methods.selectors.scissorhands import select_scissorhands_tokens
from opentome.models.mistral import patch_mistral_model, unpatch_mistral_model


def make_config(method):
    return KVCompressionConfig(
        method=method,
        max_capacity_prompt=8,
        window_size=3,
        kernel_size=3,
        sink_size=2,
        pyramid_beta=0.5,
        num_hidden_layers=2,
        quest_page_size=2,
        nacl_proxy_size=2,
        nacl_random_budget=1,
        random_seed=7,
    )


class KVCacheFactoryMethodTest(unittest.TestCase):
    def test_all_registered_policies_prefill_and_decode(self):
        torch.manual_seed(0)
        for method in POLICY_REGISTRY:
            with self.subTest(method=method):
                cache = CompressedDynamicCache(make_config(method))
                keys = torch.randn(1, 2, 12, 8)
                values = torch.randn_like(keys)
                queries = torch.randn(1, 4, 12, 8)
                attention_keys, _ = cache.update(
                    keys, values, 0, {"query_states": queries}
                )
                self.assertEqual(attention_keys.shape[-2], 12)
                cache.update(
                    torch.randn(1, 2, 1, 8),
                    torch.randn(1, 2, 1, 8),
                    0,
                    {"query_states": torch.randn(1, 4, 1, 8)},
                )
                self.assertEqual(cache.get_logical_length(), 13)
                self.assertGreater(cache.get_seq_length(), 0)

    def test_quest_selection_is_query_aware_and_keeps_recent(self):
        keys = torch.tensor(
            [[[[4.0, 0.0], [3.0, 1.0], [-1.0, 5.0], [0.0, 6.0], [9.0, 9.0]]]]
        )
        indices = select_quest_tokens(
            torch.tensor([[[0.0, 1.0]]]), keys, page_size=2,
            token_budget=3, recent_size=1,
        )
        torch.testing.assert_close(indices, torch.tensor([[[2, 3, 4]]]))

    def test_nacl_selection_protects_proxy_tokens(self):
        scores = torch.tensor([[[0.1, 0.9, 0.2, 0.8, 0.3, 0.4]]])
        proxy = select_nacl_proxy_indices(6, proxy_size=2)
        indices = select_nacl_tokens(scores, token_budget=4, proxy_indices=proxy, random_budget=0)
        torch.testing.assert_close(indices, torch.tensor([[[1, 3, 4, 5]]]))

    def test_scissorhands_protects_sink_and_recent_tokens(self):
        importance = torch.tensor([[[0.1, 0.9, 0.2, 0.8, 0.3, 0.4]]])
        indices = select_scissorhands_tokens(
            importance, token_budget=4, sink_size=1, recent_size=1
        )
        torch.testing.assert_close(indices, torch.tensor([[[0, 1, 3, 5]]]))

    def test_minicache_slerp_and_restore(self):
        previous = torch.tensor([[[[1.0, 0.0], [0.0, 2.0], [2.0, 0.0]]]])
        current = torch.tensor([[[[0.0, 3.0], [4.0, 0.0], [0.0, 2.0]]]])
        shared, _, _, angle = minicache_slerp(current, previous, interpolation=0.5)
        torch.testing.assert_close(shared.norm(dim=-1), torch.ones_like(angle))
        packed = compress_minicache_pair(current, previous, retention_count=1)
        restored_current, restored_previous = restore_minicache_pair(
            packed[0], packed[1], packed[2],
            current_retained=packed[4], previous_retained=packed[5],
            retained_indices=packed[6],
        )
        torch.testing.assert_close(restored_current.norm(dim=-1), current.norm(dim=-1))
        torch.testing.assert_close(restored_previous.norm(dim=-1), previous.norm(dim=-1))

    def test_tiny_mistral_all_registered_policies(self):
        torch.manual_seed(1)
        for method in POLICY_REGISTRY:
            with self.subTest(method=method):
                model = MistralForCausalLM(
                    MistralConfig(
                        vocab_size=64,
                        hidden_size=32,
                        intermediate_size=64,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        max_position_embeddings=64,
                        sliding_window=64,
                    )
                ).eval()
                patch_mistral_model(model)
                cache = CompressedDynamicCache(make_config(method))
                input_ids = torch.randint(0, 64, (1, 12))
                with torch.inference_mode():
                    output = model(
                        input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        past_key_values=cache,
                        use_cache=True,
                    )
                    model(
                        output.logits[:, -1].argmax(-1, keepdim=True),
                        attention_mask=torch.ones(1, 13, dtype=torch.long),
                        position_ids=torch.tensor([[12]]),
                        past_key_values=cache,
                        use_cache=True,
                    )
                self.assertEqual(cache.get_logical_length(), 13)
                unpatch_mistral_model(model)


if __name__ == "__main__":
    unittest.main()
