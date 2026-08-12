# KV Cache Inference Evaluations

The inference evaluations share `CompressedDynamicCache`, the compression
method registry, and instance-local Llama/Mistral adapters.

All default outputs are anchored at the OpenToMe repository root:

```text
work_dirs/benchmark_kv/<model-name>/<method>/<max-capacity>/result.json
work_dirs/ruler/<model-name>/<method>/<max-capacity>/predictions.jsonl
work_dirs/longbench/<model-name>/<method>/<max-capacity>/...
```

The `max-capacity` directory is the value passed through
`--max-capacity-prompt`, so runs with different KV cache budgets never resume
from or overwrite one another. An explicit `--output` path still takes
precedence for the benchmark and RULER commands.

## Latency And Memory

```bash
python -m evaluations.inference.benchmark_kv \
  --model-path /path/to/model --method snapkv \
  --max-capacity-prompt 512 --max-new-tokens 128
```

The result includes latency, generation throughput, CUDA peak memory, physical
cache bytes, and per-layer cache lengths.

## RULER

```bash
python -m evaluations.inference.ruler \
  --model-path /path/to/model --method nacl \
  --data-file /path/to/RULER/4096/niah_single_1.jsonl
```

RULER records must contain `input` and `outputs`; `index` and `length` are
preserved when available. Existing output lines are resumed unless
`--overwrite` is used.

LongBench has its runner and scorer under `longbench/`. Needle evaluation is
kept under `evaluations/needle/`. Every inference experiment writes metadata
containing arguments, library versions, timestamp, and git commit.
