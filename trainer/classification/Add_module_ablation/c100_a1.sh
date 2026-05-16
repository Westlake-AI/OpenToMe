#!/bin/bash
# ======================================================================
# A1 — DTEM (learnable metric + soft merge) variant of A0
# Identical to A0 (CLSToMEHybridModel / tomevit_small_cls) EXCEPT:
#   - Local merge: DTEM (learnable metric_layers + DTEMMergeOnly soft merge)
#     instead of ToME hard bipartite matching
#   - Adds: metric_layers (4× Linear(384→64)) + DTEMMergeOnly block
#   - --use_softkmax: ThreTopK parallel k-hot selection, replacing the
#     sequential softmax loop that causes gradient explosion at k=36
#
# Ablation chain:  A0 (ToME hard merge) → A1 (DTEM soft merge) → A5 (full MergeNet)
# A1 isolates DTEM soft merge vs ToME; patch_size / LocalBlock / LatentEncoder
# all stay identical to A0.
# ======================================================================
# bash c100_a1.sh 2>&1 | tee train_log_A1_$(date +%Y%m%d_%H%M%S).txt

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_A1_patch16_purelatent_softkmax

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node 6 \
  "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir ${DATA_DIR} \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model additive_dtem_small_cls \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 16 \
  --lambda_local 4.0 \
  --total_merge_latent 0 \
  --local_block_window 16 \
  --dtem_window_size 7 \
  --dtem_t 1 \
  --use_softkmax \
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
