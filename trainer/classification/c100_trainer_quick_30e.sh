#!/bin/bash
# bash c100_trainer_quick_30e.sh 2>&1 | tee train_log_$(date +%Y%m%d_%H%M%S).txt

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/lisiyuan/jx/.cache/cifar100
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_mergenet_small_quick30e_111

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node 8 "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir ${DATA_DIR} \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model hybridtomevit_small_cls \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 8 \
  --dtem_t 2 \
  --dtem_feat_dim 64 \
  --lambda_local 4.0 \
  --total_merge_latent 0 \
  --use_softkmax \
  --swa_size 256 \
  --batch_size 64 \
  --epochs 30 \
  --lr 5e-4 \
  --lr_local 5e-4 \
  --weight_decay 0.05 \
  --dtem_window_size 16 \
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
  --seed 42 \
