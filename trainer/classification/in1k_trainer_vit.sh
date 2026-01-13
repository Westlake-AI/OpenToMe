#!/bin/bash
# bash in1k_trainer_vit.sh 2>&1 | tee vit_train_log_$(date +%Y%m%d_%H%M%S).txt
# Training script for vanilla Vision Transformer (ViT-Base/16)

DATA_DIR=/ssdwork/yuchang/ImageNet
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=vit_base_patch16_baseline

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 in1k_trainer.py \
  --data_dir ${DATA_DIR} \
  --dataset ImageFolder \
  --train_split train \
  --val_split val \
  --model vit_base_patch16_224 \
  --num_classes 1000 \
  --batch_size 512 \
  --epochs 300 \
  --lr 5e-4 \
  --weight_decay 0.05 \
  --sched cosine \
  --warmup_epochs 5 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --aa rand-m9-mstd0.5-inc1 \
  --workers 8 \
  --amp \
  --output ${OUTPUT_DIR} \
  --experiment ${EXP_NAME} \
  --seed 42

