#!/usr/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

export BACKBONE=transformer++_340M
export TOKENIZER_NAME=blt
echo $BACKBONE
echo $TOKENIZER_NAME

MODEL='/yuchang/lsy_jx/OpenToMe/trainer/flame/exp/byte/transformers_batch1.seqlen32768.grad_acc4.warmup2048.update1.steps20480.4gpus.lr1e-3'

python -m harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 32 \
    --num_fewshot 0 \
    --device cuda \
    --show_config
