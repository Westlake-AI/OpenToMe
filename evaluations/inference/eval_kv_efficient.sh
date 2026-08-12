#!/usr/bin/bash

MODEL_PATH=$1
METHOD=$2
MAX_CAPACITY=$3

python benchmark_kv.py \
    --model-path $MODEL_PATH \
    --method $METHOD \
    --max-capacity-prompt $MAX_CAPACITY \
    --prompt-repeats 100 \
    --repeat 5