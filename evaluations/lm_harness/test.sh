#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export HF_ENDPOINT=https://hf-mirror.com
export BACKBONE=delta_net
export TOKENIZER_NAME=default

MODEL="/lisiyuan/jx/.cache/delta_net-1.3B-100B"

cd "$SCRIPT_DIR"
python -m harness --model hf \
    --model_args pretrained="$MODEL",dtype=bfloat16 \
    --tasks lambada_openai \
    --batch_size 1 \
    --num_fewshot 0 \
    --device cuda \
    --output_path "$REPO_ROOT/work_dirs/lm_harness/test" \
    --show_config
