#!/bin/bash
# Common settings for fair CIFAR-100 comparison.

export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/yuchang/yk/data
OUTPUT_DIR=./work_dirs/classification

COMMON_ARGS=(
  --data_dir "${DATA_DIR}"
  --dataset CIFAR100
  --train_split train
  --val_split val
  --num_classes 100
  --img_size 224
  --batch_size 50
  --epochs 200
  --lr 5e-4
  --weight_decay 0.05
  --sched cosine
  --warmup_epochs 20
  --mixup 0.8
  --cutmix 1.0
  --smoothing 0.1
  --aa rand-m9-mstd0.5-inc1
  --workers 32
  --amp
  --seed 42
  --pretrained
  --output "${OUTPUT_DIR}"
)
