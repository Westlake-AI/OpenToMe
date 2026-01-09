#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com

export BACKBONE=gated_deltanet
echo $BACKBONE

MODEL='/yuchang/lsy_jx/OpenToMe/trainer/flame/exp/gated_deltanet_h_340M-10B/batch1.seqlen32768.grad_acc4.warmup2048.update1.steps20480.4gpus.lr4e-4'


python -m harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks longbench \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --show_config