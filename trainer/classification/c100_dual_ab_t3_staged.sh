#!/bin/bash
# P1 第二步 — T3：分段训练 — epoch < dual_stage_b_start_epoch 仅训分支 A，之后 joint（L_A + λ L_B + γ L_fused）
# 默认前 100 epoch 只做 A；可按需改 --dual_stage_b_start_epoch。
#
# 修订（2026-05-05，对应 20260505_视觉MergeNet_P0P1P2进度与计划报告.md §3 / §4.2）：
#   §3.1 修复：开启 --dual_fused_loss_weight 0.2（仅在阶段 2 active='both' 时进 loss）；
#               同时依赖 model.py fusion_head 恒等初始化 + in1k_trainer.py 验证阶段
#               按 epoch 动态选 active_branch（阶段 1 评估走 'a'，阶段 2 走 'both'），
#               这是首轮 T3 验证全程 ~1% 的根因，三处缺一不可。
#   §3.2 修复：BRANCH_A_CKPT 默认指向 cifar100_branch_a_deit_aligned 的 model_best；
#               热启 + staged 课程能避免阶段 2 切入时把分支 A 已学特征震掉。
#   §4.2.6：实验目录加 _v2 后缀，与首轮失败实验区分落盘。
#
# bash c100_dual_ab_t3_staged.sh 2>&1 | tee train_log_dual_t3_$(date +%Y%m%d_%H%M%S).txt

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_dual_ab_t3_staged_v2
BRANCH_A_CKPT="${BRANCH_A_CKPT:-./work_dirs/classification/cifar100_branch_a_deit_aligned/model_best.pth.tar}"
EXTRA_CKPT=()
if [[ -n "$BRANCH_A_CKPT" && -f "$BRANCH_A_CKPT" ]]; then
  EXTRA_CKPT=(--branch_a_checkpoint "$BRANCH_A_CKPT")
  echo "[T3 staged] warm-start branch A from: $BRANCH_A_CKPT"
else
  echo "[T3 staged][WARN] BRANCH_A_CKPT not found ($BRANCH_A_CKPT); falling back to cold start. " \
       "Per report §3.2 / §4.2.4 this is expected to fail — set BRANCH_A_CKPT before re-running."
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir ${DATA_DIR} \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model hybridtomevit_small_cls_dual_ab \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 16 \
  --dtem_t 1 \
  --metric_grad_scale 0.1 \
  --branch_b_lambda_local 4.0 \
  --branch_b_total_merge_latent 0 \
  --fusion_type cat_linear \
  --dual_branch_train_mode staged \
  --dual_stage_b_start_epoch 100 \
  --dual_branch_loss_weight 1.0 \
  --dual_fused_loss_weight 0.2 \
  "${EXTRA_CKPT[@]}" \
  --batch_size 50 \
  --epochs 200 \
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
