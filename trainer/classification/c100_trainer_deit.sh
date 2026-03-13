#!/bin/bash
# bash c100_trainer_deit.sh 2>&1 | tee train_log_$(date +%Y%m%d_%H%M%S).txt
# --dtem_window_size None \

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "${SCRIPT_DIR}/c100_trainer_common.sh"

EXP_NAME=cifar100_deit_s_extend_lr5e4

MODEL_ARGS=(
  --model deit_s_extend
  --patch_size 16
  --drop_rate 0.0
  --attn_drop_rate 0.0
  --drop_path_rate 0.1
  --pretrained_type deit
)

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 "${SCRIPT_DIR}/in1k_trainer.py" \
  "${COMMON_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --experiment "${EXP_NAME}"

# 3.6M 1215