#!/usr/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

export BACKBONE=transformer++_340M
export TOKENIZER_NAME=blt
echo $BACKBONE
echo $TOKENIZER_NAME

MODEL=MODEL/PATH

python -m harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks lambada_openai \
    --batch_size 16 \
    --num_fewshot 0 \
    --device cuda \
    --show_config
