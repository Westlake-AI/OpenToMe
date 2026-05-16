#!/bin/bash
# temp.sh — MergeNet 修复版训练脚本
# 代码修复已应用于：
#   1. model.py: cross-attention out_proj 零初始化（消除冷启动噪声）
#   2. dtem.py:  metric_layer 梯度放行 10%（让分类 loss 指导合并决策）
#   3. model.py: LocalEncoder metric_layers 同样梯度放行 10%
#   4. dtem.py:  移除 DTEMAttention fp32 强制转换（恢复 AMP，节省显存）
# 训练配置修复：
#   5. 2 GPU 对齐 DeiT baseline 的有效 batch size
#   6. warmup=20 与 DeiT 一致

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_mergenet_temp_fix

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 "${SCRIPT_DIR}/in1k_trainer.py" \
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
  --batch_size 50 \
  --epochs 200 \
  --lr 2e-4 \
  --lr_local 5e-4 \
  --weight_decay 0.05 \
  --dtem_window_size 8 \
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
