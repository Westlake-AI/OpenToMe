#!/bin/bash
# temp_ablation_soft_topk.sh — 消融实验：可微 Soft Top-K 选择
#
# 对比 temp.sh (hard top-k, torch.no_grad)，本脚本启用 --soft_topk：
#   1. 用 ThreTopK 生成可微的 soft 选择权重 (梯度流向所有 token 的 size)
#   2. 通过 STE 将 soft 权重施加到 gathered tokens (前向不变，反向可微)
#   3. 辅助加权池化路径：对所有 patch token 做 soft_sel 加权均值 → head → aux_logits
#      训练时: logits = main_logits + aux_weight * aux_logits
#      评估时: logits = main_logits (仅主路径)
#   → 使 75% 被丢弃 token 的特征和权重都获得来自 loss 的梯度
#
# 同时修复了 CLS 索引偏移 bug (topk_indices → topk_indices+1 用于 gather)。

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_ablation_soft_topk

CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node 2 "${SCRIPT_DIR}/in1k_trainer.py" \
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
  --soft_topk \
  --soft_topk_aux_weight 0.3 \
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
