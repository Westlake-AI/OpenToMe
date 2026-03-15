#!/usr/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

export BACKBONE=blt_380M_10B_500hash
echo $BACKBONE
export TOKENIZER_NAME=blt
echo $TOKENIZER_NAME

MODEL='/lisiyuan/jx/OpenToMe/trainer/flame/exp/blt_380M_10B_500hash/batch1.seqlen16384.grad_acc8.warmup1024.update1.steps30720.4gpus.lr3e-4'

python -m harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --show_config