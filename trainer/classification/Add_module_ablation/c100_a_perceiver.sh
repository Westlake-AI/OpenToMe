#!/bin/bash
# ======================================================================
# A-Perceiver — Cross-Attention Pooling (Perceiver Resampling) variant of A0
# Identical to A0 (CLSToMEHybridModel / tomevit_small_cls) EXCEPT:
#   - Local merge: Perceiver Resampling (learnable latent queries +
#     cross-attention pooling) instead of ToME hard bipartite matching
#   - Adds ~1.8M params (cross-attn + FFN + latent queries) vs ToME's 0
#
# Local:  4× LocalBlock (window attention, window=16)
# Merge:  Perceiver Resampling — 49 latent queries attend to all 197
#         tokens (CLS + 196 patches) via cross-attention, output
#         [CLS] + [49 queries] = 50 tokens  (same as A0 after ToME merge)
# Latent: 8× standard ViT blocks, total_merge_latent=0 (no ToME)
# All training hyper-parameters kept identical to A0.
# ======================================================================
# bash c100_a_perceiver.sh 2>&1 | tee train_log_A_perceiver_$(date +%Y%m%d_%H%M%S).txt

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_A_perceiver_patch8_purelatent

CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node 2 \
  "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir ${DATA_DIR} \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model additive_perceiver_small_cls \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 8 \
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
