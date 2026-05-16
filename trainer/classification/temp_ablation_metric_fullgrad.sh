#!/bin/bash
# temp_ablation_metric_fullgrad.sh — 消融实验：Metric Layer 梯度全放行
#
# 对比 temp.sh (metric_grad_scale=0.1)，本脚本设置 metric_grad_scale=1.0
# 使 LocalEncoder metric_layers 的输入不再 detach，分类 loss 梯度完整回传到
# 合并决策参数，验证端到端可微 metric 对训练质量的影响。
#
# 风险：梯度可能过大导致 metric_layer 不稳定，需配合 clip_grad 观察。

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_ablation_metric_fullgrad

CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node 2 "${SCRIPT_DIR}/in1k_trainer.py" \
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
  --metric_grad_scale 1.0 \
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
