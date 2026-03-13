#!/bin/bash
# HybridToMe lambda=1 完美恢复测试: 预训练微调 CIFAR-100
# lambda_local=1.0 => total_merge_local=0, 不做任何 token 压缩
# 预期结果应与纯 DeiT baseline 一致
# bash c100_finetune_ours_lambda1.sh 2>&1 | tee train_log_ours_lambda1_ft_$(date +%Y%m%d_%H%M%S).txt

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_hybridtome_small_finetune_lambda1

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir ${DATA_DIR} \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model hybridtomevit_small_cls \
  --pretrained \
  --pretrained_type deit \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 8 \
  --lambda_local 1.0 \
  --total_merge_latent 0 \
  --batch_size 50 \
  --epochs 30 \
  --lr 1e-4 \
  --lr_local 1e-4 \
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
