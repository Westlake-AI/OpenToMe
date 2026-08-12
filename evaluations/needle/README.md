# Needle In A Haystack

This is the single Needle evaluation entry point for normal inference and
OpenToMe KV cache compression. All artifacts are fixed under:

```text
work_dirs/needle/results
work_dirs/needle/contexts
work_dirs/needle/visualizations
```

The entry point can be run from any working directory:

```bash
python evaluations/needle/needle_in_haystack.py \
  --model-path /path/to/llama-or-mistral \
  --tokenizer-path /path/to/tokenizer \
  --method snapkv \
  --max-capacity-prompt 512 --window-size 32 \
  --s-len 4096 --e-len 8192 \
  --context-lengths-min 4096 --context-lengths-max 8192
```

Use `--method none` for the uncompressed baseline. Compressed runs use one-shot
prefill; `--prefilling-chunk-size` is therefore only accepted for the baseline.
Each result records physical cache bytes, physical per-layer cache lengths, and
logical cache length. The method is included in the result directory name so
compressed runs cannot overwrite baseline results. Compressed run names end in
`_<method>_<max-capacity>`, so different KV cache budgets also produce separate
result, context, and visualization paths.
