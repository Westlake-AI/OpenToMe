# KV Compression

The package has three layers:

- `base.py`: configuration and the policy interface.
- `cache.py`: Transformers-compatible compressed cache state.
- `methods/`: independent compression implementations and their registry.

The registry currently contains StreamingKV/StreamingLLM, H2O, SnapKV,
PyramidKV, L2Norm, CAM, Quest, NACL, and Scissorhands. Reusable algorithm
cores live in `methods/selectors/`; MiniCache is exposed there because it
merges adjacent layers and is not a normal per-layer cache policy.

To add a method, create a `KVCompressionPolicy` subclass in `methods/` and add
it to `POLICY_REGISTRY`, or call `register_policy` before constructing the
cache. Deleting a method only requires removing that module and registry entry.

## Capability notes

| Method | Unified cache policy | Llama | Mistral | Notes |
| --- | --- | --- | --- | --- |
| StreamingKV/StreamingLLM | yes | yes | yes | sink plus recent window |
| H2O, SnapKV, PyramidKV | yes | yes | yes | bounded physical cache |
| L2Norm, CAM | yes | yes | yes | one-shot prefill selection |
| NACL, Scissorhands | yes | yes | yes | deterministic seed is configurable |
| Quest | yes | yes | yes | full physical storage; query-aware retrieval budget |
| MiniCache | algorithm core | no | no | cross-layer integration still required |

AdaKV/HeadKV need ragged per-head cache layouts, ThinK needs key-channel-aware
attention, HeadInfer needs CPU offload plus FlashAttention, and MInference needs
its external kernels. Quantized cache methods also require newer Transformers
cache APIs and quantization backends. They are intentionally not registered as
runnable policies until those runtime contracts are implemented.

Quest's `cache_bytes` reflects full KV storage. Its budget limits the tokens
presented to attention, not the stored cache, so it should not be compared as a
physical-memory compression result.

The Quest, NACL, Scissorhands, and MiniCache selector implementations were
adapted from KVCache-Factory (MIT), revision
`94255b6fe5127117f2e7f3b6d7ca7bd155ba9ab0`.
