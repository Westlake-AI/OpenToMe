#!/bin/bash
# P1 第二步 — T2：奇偶 step 交替只反传 A 或 B（显存上只前向单分支）
#
# 修订（2026-05-05，对应 20260505_视觉MergeNet_P0P1P2进度与计划报告.md §3 / §4.2）：
#   §3.1：alternate 模式 forward 永远走 active='a' 或 'b'，不会进入 fusion_head 路径，
#         所以 --dual_fused_loss_weight 不参与 loss，首版保持默认 0.0（不显式传）。
#         注：依赖 model.py fusion_head 恒等初始化 + in1k_trainer.py 验证阶段
#         动态 active_branch（验证时仍走 'both'），二者必须先打上。
#   §4.2.4：T2 是否热启留作消融对比项。BRANCH_A_CKPT 默认仍为空（冷启动），
#           可通过环境变量手动覆写做对比实验。
#   §4.2.6：实验目录加 _v2 后缀，与首轮失败实验区分落盘。
#
# bash c100_dual_ab_t2_alternate.sh 2>&1 | tee train_log_dual_t2_$(date +%Y%m%d_%H%M%S).txt

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_dual_ab_t2_alternate_v2
BRANCH_A_CKPT="${BRANCH_A_CKPT:-}"
EXTRA_CKPT=()
if [[ -n "$BRANCH_A_CKPT" && -f "$BRANCH_A_CKPT" ]]; then
  EXTRA_CKPT=(--branch_a_checkpoint "$BRANCH_A_CKPT")
  echo "[T2 alternate] warm-start branch A from: $BRANCH_A_CKPT (ablation contrast)"
else
  echo "[T2 alternate] cold-start (default per §4.2.4 — T2 留作消融对比项)"
fi


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node 4 "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir ${DATA_DIR} \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model hybridtomevit_small_cls_dual_ab \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 8 \
  --dtem_t 1 \
  --metric_grad_scale 0.1 \
  --branch_b_lambda_local 4.0 \
  --branch_b_total_merge_latent 0 \
  --fusion_type cat_linear \
  --dual_branch_train_mode alternate \
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
