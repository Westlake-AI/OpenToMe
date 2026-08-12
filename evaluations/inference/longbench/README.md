# LongBench KV Cache Compression Evaluation

The launcher always runs prediction followed by LongBench scoring:

```bash
bash evaluations/inference/longbench/run_longbench.sh \
  /path/to/model snapkv qasper 2048 flash_attention_2
```

Its positional arguments are:

```text
MODEL_PATH [METHOD] [DATASET] [MAX_CAPACITY] [ATTN_IMPLEMENTATION]
```

Defaults are `snapkv`, `all`, `2048`, and `flash_attention_2`. The data and
output roots are fixed relative to the OpenToMe repository:

```text
data/LongBench
work_dirs/longbench
```

Predictions and scores are written to:

```text
work_dirs/longbench/<model-name>/<method>/<max-capacity>/longbench/<dataset>.jsonl
work_dirs/longbench/<model-name>/<method>/<max-capacity>/longbench/result.json
```

Existing JSONL lines are treated as completed samples. The evaluator can also
be called directly:

```bash
python evaluations/inference/longbench/evaluate.py \
  --prediction-path work_dirs/longbench/<model-name>/snapkv/2048/longbench
```

Each prediction contains the LongBench reference fields plus token counts,
generation time, physical per-layer cache lengths, and KV cache bytes. Metrics
include Rouge-L, QA F1, classification, retrieval, counting, and code
similarity. Chinese metrics use `jieba` when installed and otherwise fall back
to character tokenization.

Runnable methods are `streamingkv`/`streamingllm`, `h2o`, `snapkv`,
`pyramidkv`, `l2norm`, `cam`, `quest`, `nacl`, and `scissorhands`.
