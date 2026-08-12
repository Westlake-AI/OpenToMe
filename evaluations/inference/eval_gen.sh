#!/usr/bin/bash

# Simple evaluation script for text generation benchmarking
# Usage: ./eval.sh <model_path> <model_type> [additional_args]

MODEL_PATH=$1
MODEL_TYPE=$2
TOKENIZER=$3
PROMPT="Please introduce Westlake University in 100 words"

echo "Evaluating model: $MODEL_TYPE"
echo "Model path: $MODEL_PATH"
echo "Tokenizer: $TOKENIZER"

python benchmark_generation.py \
    --path "$MODEL_PATH" \
    --model "$MODEL_TYPE" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --output-generation \
    --repetition_penalty=2.0 \
    --temperature 0.5 \
    --topp 0.9 \
    --length 128 \
    --maxlen 256 \
    --no-cache \