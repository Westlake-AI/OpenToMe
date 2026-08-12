import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from opentome.compress import CompressedDynamicCache, KVCompressionConfig
from opentome.models.llama import patch_llama_model, unpatch_llama_model


METHODS = ("streamingkv", "h2o", "snapkv", "pyramidkv")


def make_config(method):
    return KVCompressionConfig(
        method=method,
        max_capacity_prompt=8,
        window_size=3,
        kernel_size=3,
        sink_size=2,
        pyramid_beta=0.5,
        num_hidden_layers=2,
    )


class KVCacheCompressionTest(unittest.TestCase):
    def test_compressed_cache_prefill_and_decode(self):
        torch.manual_seed(0)
        for method in METHODS:
            with self.subTest(method=method):
                cache = CompressedDynamicCache(make_config(method))
                keys = torch.randn(1, 2, 12, 8)
                values = torch.randn_like(keys)
                queries = torch.randn(1, 4, 12, 8)
                attention_keys, attention_values = cache.update(
                    keys, values, layer_idx=1, cache_kwargs={"query_states": queries}
                )
                self.assertEqual(attention_keys.shape[-2], 12)
                self.assertEqual(attention_values.shape[-2], 12)
                self.assertEqual(cache.get_seq_length(1), cache.policy.capacity_for_layer(1))
                self.assertEqual(cache.get_logical_length(1), 12)

                new_keys = torch.randn(1, 2, 1, 8)
                new_values = torch.randn_like(new_keys)
                new_queries = torch.randn(1, 4, 1, 8)
                cache.update(
                    new_keys, new_values, layer_idx=1,
                    cache_kwargs={"query_states": new_queries},
                )
                capacity = cache.policy.capacity_for_layer(1)
                expected = capacity if method in {"streamingkv", "h2o"} else capacity + 1
                self.assertEqual(cache.get_seq_length(1), expected)
                self.assertEqual(cache.get_logical_length(1), 13)

    def test_tiny_llama_gqa_generation_path(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = LlamaForCausalLM(
                    LlamaConfig(
                        vocab_size=64,
                        hidden_size=32,
                        intermediate_size=64,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        max_position_embeddings=64,
                    )
                ).eval()
                patch_llama_model(model)
                cache = CompressedDynamicCache(make_config(method))
                input_ids = torch.randint(0, 64, (1, 12))
                with torch.inference_mode():
                    outputs = model(
                        input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        past_key_values=cache,
                        use_cache=True,
                    )
                    next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
                    outputs = model(
                        next_token,
                        attention_mask=torch.ones(1, 13, dtype=torch.long),
                        position_ids=torch.tensor([[12]]),
                        past_key_values=cache,
                        use_cache=True,
                    )

                self.assertEqual(outputs.logits.shape, (1, 1, 64))
                self.assertEqual(cache.seen_tokens, 13)
                self.assertEqual(cache.get_logical_length(), 13)
                self.assertTrue(all(length > 0 for length in cache.layer_lengths()))
                unpatch_llama_model(model)
                self.assertFalse(hasattr(model.model.layers[0].self_attn, "_opentome_original_forward"))

    def test_invalid_compression_configuration(self):
        with self.assertRaisesRegex(ValueError, "window_size"):
            KVCompressionConfig(max_capacity_prompt=8, window_size=8)


if __name__ == "__main__":
    unittest.main()
