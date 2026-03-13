#!/bin/bash
# bash c100_tome.sh 2>&1 | tee train_log_$(date +%Y%m%d_%H%M%S).txt
# ToME ablation: replaces DTEM+Perceiver with pure ToME bipartite matching

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_tomenet_small_quick30e

OPENTOME_MERGENET_IMPL=tome CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir ${DATA_DIR} \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model tomevit_small_cls \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 8 \
  --lambda_local 4.0 \
  --total_merge_latent 0 \
  --batch_size 50 \
  --epochs 30 \
  --lr 5e-4 \
  --lr_local 5e-4 \
  --weight_decay 0.05 \
  --sched cosine \
  --warmup_epochs 3 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --aa rand-m9-mstd0.5-inc1 \
  --workers 32 \
  --amp \
  --output ${OUTPUT_DIR} \
  --experiment ${EXP_NAME} \
  --seed 42
