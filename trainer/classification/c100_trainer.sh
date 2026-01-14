#!/bin/bash
# bash c100_trainer.sh 2>&1 | tee train_log_$(date +%Y%m%d_%H%M%S).txt

DATA_DIR=/yuchang/lsy/data/cifar100/
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_mergenet_small_260113

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 in1k_trainer.py \
  --data_dir ${DATA_DIR} \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model hybridtomevit_small_cls \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 8 \
  --dtem_window_size 7 \
  --dtem_r 4 \
  --dtem_t 2 \
  --dtem_feat_dim 64 \
  --lambda_local 4.0 \
  --total_merge_latent 0 \
  --num_local_blocks 1 \
  --local_block_window 32 \
  --batch_size 100 \
  --epochs 200 \
  --lr 3e-4 \
  --weight_decay 0.05 \
  --sched cosine \
  --warmup_epochs 20 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --aa rand-m9-mstd0.5-inc1 \
  --workers 8 \
  --amp \
  --output ${OUTPUT_DIR} \
  --experiment ${EXP_NAME} \
  --seed 42 \
  --use_softkmax
