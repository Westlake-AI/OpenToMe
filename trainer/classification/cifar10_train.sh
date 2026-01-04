#!/bin/bash

DATA_DIR=/yuchang/yk/opentome_new/data
OUTPUT_DIR=./work_dirs/classification_cifar10
EXP_NAME=cifar10_test

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 in1k_trainer.py \
  --data_dir ${DATA_DIR} \
  --dataset torch/CIFAR10 \
  --train_split train \
  --val_split test \
  --model hybridtomevit_base \
  --num_classes 10 \
  --dtem_window_size 8 \
  --dtem_r 4 \
  --dtem_t 2 \
  --total_merge_local 98 \
  --total_merge_latent 4 \
  --use_softkmax \
  --use_cross_attention \
  --img_size 224 \
  --batch_size 64 \
  --epochs 200 \
  --lr 5e-5 \
  --warmup_lr 1e-5 \
  --weight_decay 0.05 \
  --sched cosine \
  --warmup_epochs 5 \
#   --clip_grad 1.0 \
#   --clip_mode norm \
  --mixup 0.2 \
  --cutmix 0.0 \
  --smoothing 0.1 \
  --workers 4 \
  --output ${OUTPUT_DIR} \
  --experiment ${EXP_NAME} \
  --seed 42
