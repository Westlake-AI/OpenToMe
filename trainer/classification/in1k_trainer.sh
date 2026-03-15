#!/bin/bash
# bash c100_trainer_quick_30e.sh 2>&1 | tee train_log_$(date +%Y%m%d_%H%M%S).txt

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/lisiyuan/.cache/imagenet
OUTPUT_DIR=./work_dirs/imagenet
EXP_NAME=imagenet_mergenet_small

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir ${DATA_DIR} \
  --dataset ImageNet \
  --train_split train \
  --val_split val \
  --model hybridtomevit_small_cls \
  --num_classes 1000 \
  --img_size 224 \
  --patch_size 8 \
  --dtem_t 2 \
  --dtem_feat_dim 64 \
  --lambda_local 4.0 \
  --total_merge_latent 0 \
  --use_softkmax \
  --swa_size 256 \
  --batch_size 256 \
  --epochs 300 \
  --lr 1e-3 \
  --lr_local 1e-3 \
  --weight_decay 0.05 \
  --dtem_window_size 8 \
  --sched cosine \
  --warmup_epochs 5 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --aa rand-m9-mstd0.5-inc1 \
  --workers 32 \
  --amp \
  --output ${OUTPUT_DIR} \
  --experiment ${EXP_NAME} \
  --seed 42 \
  --resume ./work_dirs/imagenet/imagenet_mergenet_small/checkpoint-132.pth.tar