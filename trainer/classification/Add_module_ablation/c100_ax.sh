#!/bin/bash
# ======================================================================
# AX — Local: timm ViT blocks + tome_apply_patch (ToMeBlock, 4 layers)
# Latent: 8× standard ViT blocks, NO ToME (total_merge_latent=0 → use_tome=False)
# Compare with A0 (LocalBlock + post-hoc merge + same pure latent after fix)
# ======================================================================
# bash c100_ax.sh 2>&1 | tee train_log_AX_$(date +%Y%m%d_%H%M%S).txt

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_AX_patch16

OPENTOME_MERGENET_IMPL=tome \
CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node 2 \
  "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir ${DATA_DIR} \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model tomevit_small_cls_ax \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 16 \
  --lambda_local 4.0 \
  --total_merge_latent 0 \
  --local_block_window 16 \
  --dtem_window_size 7 \
  --dtem_t 1 \
  --batch_size 50 \
  --epochs 200 \
  --lr 1e-3 \
  --lr_local 1e-3 \
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
