#!/bin/bash
# 纯 DeiT-Small baseline: 预训练微调 CIFAR-100
# bash c100_finetune_deit.sh 2>&1 | tee train_log_deit_ft_$(date +%Y%m%d_%H%M%S).txt

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_deit_small_finetune_pretrained

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir ${DATA_DIR} \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model deit_small_patch16_224 \
  --pretrained \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 8 \
  --batch_size 50 \
  --epochs 30 \
  --lr 1e-4 \
  --weight_decay 0.05 \
  --sched cosine \
  --warmup_epochs 3 \
  --mixup 0.0 \
  --cutmix 0.0 \
  --smoothing 0.1 \
  --aa rand-m9-mstd0.5-inc1 \
  --workers 32 \
  --amp \
  --output ${OUTPUT_DIR} \
  --experiment ${EXP_NAME} \
  --seed 42
