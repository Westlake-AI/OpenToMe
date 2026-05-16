#!/bin/bash
# ======================================================================
# A5 终点 — CLSHybridToMeModel (hybridtomevit_small_cls)
# Local: DTEM soft merge | topk + center-of-mass sort | encode cross-attn
# Latent: LatentEncoder(ToME) | Forward: full MergeNet pipeline
# ======================================================================
# bash c100_a5.sh 2>&1 | tee train_log_A5_$(date +%Y%m%d_%H%M%S).txt

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_A5

CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node 2 \
  "${SCRIPT_DIR}/in1k_trainer.py" \
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
  --dtem_window_size 16 \
  --lambda_local 4.0 \
  --total_merge_latent 0 \
  --use_softkmax \
  --batch_size 50 \
  --epochs 200 \
  --lr 1e-3 \
  --lr_local 1e-3 \
  --weight_decay 0.05 \
  --sched cosine \
  --clip_grad 1.0 \
  --warmup_epochs 5 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --aa rand-m9-mstd0.5-inc1 \
  --workers 32 \
  --amp \
  --output ${OUTPUT_DIR} \
  --experiment ${EXP_NAME} \
  --seed 42
