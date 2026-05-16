#!/bin/bash
# 仅 Stage2：hybridtomevit_small_cls_dual_ab（joint + fused），随机初始化，不加载任何 checkpoint / 预训练。
#
# 用法：
#   bash 1000s2.sh 2>&1 | tee train_log_stage2_scratch_1000e_$(date +%Y%m%d_%H%M%S).txt
#
# 可选环境变量：
#   CUDA_VISIBLE_DEVICES=4,5,6,7
#   NPROC_PER_NODE=4
#   DATA_DIR=/path/to/data
#   OUTPUT_DIR=./work_dirs/classification
#   EXP_PREFIX=cifar100_stage2_scratch_p8C
#   STAGE2_EPOCHS=1000
#   FREEZE_BRANCH_A_UNTIL=0              # >0 时启用 branch_a lr_scale 冻结+ramp（与 in1k_trainer 一致）
#   BRANCH_A_LR_RAMP=5

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR="${DATA_DIR:-/liziqing/yukai/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./work_dirs/classification}"
EXP_PREFIX="${EXP_PREFIX:-cifar100_stage2_scratch_p8C}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-1000}"
FREEZE_BRANCH_A_UNTIL="${FREEZE_BRANCH_A_UNTIL:-0}"
BRANCH_A_LR_RAMP="${BRANCH_A_LR_RAMP:-5}"

if [[ "${FREEZE_BRANCH_A_UNTIL}" -gt 0 ]]; then
  STAGE2_TAG="lr0until${FREEZE_BRANCH_A_UNTIL}_ramp${BRANCH_A_LR_RAMP}_cliplocal"
else
  STAGE2_TAG="scratch_nofreeze"
fi
STAGE2_EXP="${EXP_PREFIX}_stage2_merge_${STAGE2_EPOCHS}e_${STAGE2_TAG}"

mkdir -p "${OUTPUT_DIR}"

EXTRA_FREEZE=()
if [[ "${FREEZE_BRANCH_A_UNTIL}" -gt 0 ]]; then
  EXTRA_FREEZE=(
    --freeze_branch_a_until_epoch "${FREEZE_BRANCH_A_UNTIL}"
    --branch_a_lr_ramp_epochs "${BRANCH_A_LR_RAMP}"
  )
  echo "[Stage2] branch_a lr_scale schedule: 0 until epoch ${FREEZE_BRANCH_A_UNTIL}, ramp ${BRANCH_A_LR_RAMP} epochs"
else
  echo "[Stage2] no branch_a freeze schedule (FREEZE_BRANCH_A_UNTIL=0)"
fi

echo "[Stage2] dual_ab from scratch (no --pretrained, no --branch_a_checkpoint), ${STAGE2_EPOCHS} epochs → ${STAGE2_EXP}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" "${SCRIPT_DIR}/in1k_trainer.py" \
  --data_dir "${DATA_DIR}" \
  --dataset CIFAR100 \
  --train_split train \
  --val_split val \
  --model hybridtomevit_small_cls_dual_ab \
  --num_classes 100 \
  --img_size 224 \
  --patch_size 8 \
  --dtem_t 1 \
  --dtem_feat_dim 64 \
  --metric_grad_scale 0.1 \
  --branch_b_lambda_local 4.0 \
  --branch_b_total_merge_latent 0 \
  --branch_b_dtem_window_size 8 \
  --branch_b_use_softkmax \
  --branch_b_swa_size 256 \
  --fusion_type cat_linear \
  --dual_branch_train_mode joint \
  --dual_branch_loss_weight 1.0 \
  --dual_fused_loss_weight 0.2 \
  "${EXTRA_FREEZE[@]}" \
  --batch_size 50 \
  --epochs "${STAGE2_EPOCHS}" \
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
  --output "${OUTPUT_DIR}" \
  --experiment "${STAGE2_EXP}" \
  --seed 42

echo "[Done] stage2=${OUTPUT_DIR}/${STAGE2_EXP}"
