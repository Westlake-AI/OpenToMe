#!/usr/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

export BACKBONE=gsa_340M
echo $BACKBONE
export TOKENIZER_NAME=blt
echo $TOKENIZER_NAME

MODEL='/yuchang/lsy_jx/.cache/opetome_ckpt/check_ok/byte/gsa-340M-10B/batch1.seqlen32768.grad_acc4.warmup1024.update1.steps30720.4gpus.lr3e-4'

python -m harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --show_config