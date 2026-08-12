#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
OUTPUT_DIR="$ROOT/work_dirs/dna_hg38"
PYTHON=${PYTHON:-/opt/conda/envs/fla/bin/python}
MAX_LENGTH=${MAX_LENGTH:-32768}
DEVICE=${DEVICE:-cuda}
SPLIT=${SPLIT:-test}

mkdir -p "$OUTPUT_DIR"

models=(
  gla_dna_54M_32k
  hyenadna_small_32k
  delta_net_dna_54M_32k
  gated_deltanet_dna_54M_32k
  transformer_dna_54M_32k
)

model_tag=hg38.seqlen_schedule0:32k,2:64k,5:128k.epochs5.global_batch256.warmup1pct.4gpus.lr1e-3/step-665

for model in "${models[@]}"; do
  echo "Evaluating ${model} on ${SPLIT}, max_length=${MAX_LENGTH}"
  "${PYTHON}" "$SCRIPT_DIR/hyenadna_hg38_ppl.py" \
    --model_dir "$ROOT/trainer/flame/exp/dna/${model}/${model_tag}" \
    --fasta "$ROOT/data/hg38/hg38.ml.fa.gz" \
    --bed "$ROOT/data/hg38/human-sequences.bed" \
    --split "$SPLIT" \
    --max_length "$MAX_LENGTH" \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/${model}_${SPLIT}_ppl_${MAX_LENGTH}.json"
done
