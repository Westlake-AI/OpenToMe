#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

MODEL_PATH=${1:?"Usage: $0 MODEL_PATH [METHOD] [DATASET] [MAX_CAPACITY] [ATTN_IMPLEMENTATION]"}
METHOD=${2:-snapkv}
DATASET=${3:-all}
MAX_CAPACITY=${4:-2048}
ATTN_IMPLEMENTATION=${5:-flash_attention_2}
DATASET_PATH="$REPO_ROOT/data/LongBench"
OUTPUT_DIR="$REPO_ROOT/work_dirs/longbench"
RUN_NAME=$(basename -- "${MODEL_PATH%/}")
PREDICTION_DIR="$OUTPUT_DIR/$RUN_NAME/$METHOD/$MAX_CAPACITY/longbench"

python "$SCRIPT_DIR/predict.py" \
    --model-path "$MODEL_PATH" \
    --method "$METHOD" \
    --dataset "$DATASET" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --max-capacity-prompt "$MAX_CAPACITY" \
    --attn-implementation "$ATTN_IMPLEMENTATION"

EVAL_ARGS=(--prediction-path "$PREDICTION_DIR")
if [[ "$DATASET" != "all" ]]; then
    EVAL_ARGS+=(--dataset "$DATASET")
fi
python "$SCRIPT_DIR/evaluate.py" "${EVAL_ARGS[@]}"
