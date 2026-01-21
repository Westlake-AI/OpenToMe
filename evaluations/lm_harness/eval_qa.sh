#!/usr/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

export BACKBONE=gated_deltanet_340M
export TOKENIZER_NAME=blt
echo $BACKBONE
echo $TOKENIZER_NAME

MODEL='/masiqi/lisiyuan/jx/OpenToMe/trainer/flame/exp/byte/gated_deltanet_340M-10B/batch1.seqlen32768.grad_acc4.warmup1024.update1.steps30720.4gpus.lr4e-4'

python -m harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 32 \
    --num_fewshot 0 \
    --device cuda \
    --show_config
