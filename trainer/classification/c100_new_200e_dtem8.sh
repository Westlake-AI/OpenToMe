#!/bin/bash
# bash c100_new_200e_dtem8.sh 2>&1 | tee train_log_$(date +%Y%m%d_%H%M%S).txt
#
# 单阶段 mergenet (HybridToMe) 对照组，用于与 c100_a200e_then_merge200e.sh
# (stage1 200e branch_a + stage2 200e dual_ab merge, 总预算 400 epochs)
# 做"同训练预算下方法 vs 单阶段 mergenet"的公平对比。
#
# 与 c100_a200e_then_merge200e.sh 严格对齐的项：
#   - 数据/增强/seed:        CIFAR100 / mixup0.8 / cutmix1.0 / smoothing0.1 / aa rand-m9 / seed 42
#   - 优化器协议:            AdamW(lr 1e-3, wd 0.05, clip 1.0), cosine, warmup 20e
#   - 硬件 / effective batch: 4 GPU (4,5,6,7) × batch_size 50 = 200
#   - 总训练预算:            400 epochs (= 200 stage1 + 200 stage2)
#   - patch_size:            8  (token 数 28*28=784 与 stage1 一致)
#   - amp / workers / num_classes / img_size 一致
#
# mergenet 方法本身的差异（不属于"协议对齐"，是要对比的设计本身，保持不变）：
#   - 完整 MergeNet 路径: dtem_t=2, dtem_feat_dim=64, dtem_window_size=8,
#     use_softkmax, swa_size=256, lambda_local=4.0, total_merge_latent=0
#   - 注意 lambda_local=4.0 ≠ stage1 branch_a 的 1.0，这正是 mergenet 在做空间压缩

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_mergenet_small_400e_dtem8_b200

# OPENTOME_MERGENET_IMPL 不设置则默认 new
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node 4 "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir ${DATA_DIR} \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model hybridtomevit_small_cls \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 8 \
  --dtem_t 1 \
  --dtem_feat_dim 64 \
  --lambda_local 4.0 \
  --total_merge_latent 0 \
  --use_softkmax \
  --swa_size 256 \
  --dtem_window_size 8 \
  --batch_size 50 \
  --epochs 400 \
  --lr 1e-3 \
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
