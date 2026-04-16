#!/usr/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

export BACKBONE=delta_net
export TOKENIZER_NAME=default
echo $BACKBONE
echo $TOKENIZER_NAME

MODEL="/lisiyuan/jx/.cache/delta_net-1.3B-100B"

python -m harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks lambada_openai \
    --batch_size 1 \
    --num_fewshot 0 \
    --device cuda \
    --show_config
