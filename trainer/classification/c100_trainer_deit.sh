#!/bin/bash
# bash c100_trainer.sh 2>&1 | tee train_log_$(date +%Y%m%d_%H%M%S).txt
# --dtem_window_size None \

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

DATA_DIR=/liziqing/lisiyuan/jx/.cache/cifar100
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_deit_s_extend_lr5e4


CUDA_VISIBLE_DEVICES=4,5 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nproc_per_node 2 "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir ${DATA_DIR} \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model deit_s_extend \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 16 \
  --batch_size 50 \
  --epochs 200 \
  --lr 5e-4 \
  --weight_decay 0.05 \
  --drop_rate 0.0 \
  --attn_drop_rate 0.0 \
  --drop_path_rate 0.1 \
  --sched cosine \
  --warmup_epochs 20 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --aa rand-m9-mstd0.5-inc1 \
  --workers 32 \
  --amp \
  --output ${OUTPUT_DIR} \
  --experiment ${EXP_NAME} \
  --seed 42 \
  --pretrained \
  --pretrained_type deit