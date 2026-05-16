#!/bin/bash
# 双分支路线 P1 — 第一步：分支 A 单独训练，与 c100_deit_200e.sh 数据/增强/epoch/优化器一致，仅换模型。
# 模型：hybridtomevit_small_cls_branch_a（lambda_local=1、total_merge_latent=0，无降采样 MergeNet 路径）
#
# bash c100_branch_a_deit_aligned.sh 2>&1 | tee train_log_branch_a_$(date +%Y%m%d_%H%M%S).txt

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_branch_a_deit_aligned

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node 8 "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir ${DATA_DIR} \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model hybridtomevit_small_cls_branch_a \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 16 \
  --batch_size 50 \
  --epochs 200 \
  --lr 1e-3 \
  --weight_decay 0.05 \
  --sched cosine \
  --clip_grad 1.0 \
  --warmup_epochs 20 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --aa rand-m9-mstd0.5-inc1 \
  --workers 32 \
  --amp \
  --output ${OUTPUT_DIR} \
  --experiment ${EXP_NAME} \
  --seed 42
