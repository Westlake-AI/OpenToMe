#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export HF_ENDPOINT=https://hf-mirror.com
export BACKBONE=blt_380M_10B_500hash
export TOKENIZER_NAME=blt

MODEL='/lisiyuan/jx/OpenToMe/trainer/flame/exp/blt_380M_10B_500hash/batch1.seqlen16384.grad_acc8.warmup1024.update1.steps30720.4gpus.lr3e-4'

cd "$SCRIPT_DIR"
python -m harness --model hf \
    --model_args pretrained="$MODEL",dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --output_path "$REPO_ROOT/work_dirs/lm_harness/qa_byte" \
    --show_config
