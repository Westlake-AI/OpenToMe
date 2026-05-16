#!/bin/bash
# P1 第二步 — T1：双分支同时从零并联训练，loss = L_A + λ L_B + γ L_fused
#
# 修订（2026-05-05，对应 20260505_视觉MergeNet_P0P1P2进度与计划报告.md §3 / §4.2）：
#   §3.1 修复：开启 --dual_fused_loss_weight 0.2，让 fusion_head 收到显式监督；
#               (配合 model.py 中 fusion_head 恒等到平均融合初始化 +
#                in1k_trainer.py 验证阶段动态 active_branch 选择，三处缺一不可)
#   §4.2.6：实验目录加 _v2 后缀，与首轮失败实验区分落盘。
#
# 角色定位（用户 2026-05-05 决议）：
#   T1 默认 *冷启动*，作为「不依赖 P1 第一步 ckpt」的对照实验。CC
#   - T3 staged_v2 (策略 B)：热启 + 联合从 epoch 0 → 主推路径
#   - T1 joint_v2     ：从零并联训双分支             → 检验"无热启 joint 是否可行"
#   - T2 alternate_v2 ：从零、奇偶 step 交替单分支    → 检验"显存 1× 路径是否可行"
#   三者构成正交对照，不再让 T1 与 T3 在初始化上重复。
#   如需把 T1 也做"热启版"消融，临时设置环境变量即可：
#     BRANCH_A_CKPT=./work_dirs/classification/cifar100_branch_a_deit_aligned/model_best.pth.tar \
#         bash c100_dual_ab_t1_joint.sh
#
# bash c100_dual_ab_t1_joint.sh 2>&1 | tee train_log_dual_t1_$(date +%Y%m%d_%H%M%S).txt

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/liziqing/yuhao/yukai/data
OUTPUT_DIR=./work_dirs/classification
EXP_NAME=cifar100_dual_ab_t1_joint_v2
BRANCH_A_CKPT="${BRANCH_A_CKPT:-}"

EXTRA_CKPT=()
if [[ -n "$BRANCH_A_CKPT" && -f "$BRANCH_A_CKPT" ]]; then
  EXTRA_CKPT=(--branch_a_checkpoint "$BRANCH_A_CKPT")
  echo "[T1 joint] warm-start branch A from: $BRANCH_A_CKPT (override via env)"
else
  echo "[T1 joint] cold-start (default) — both branches trained from scratch. T3-staged_v2 是热启对照路径"
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node 4 "${SCRIPT_DIR}/in1k_trainer.py" \
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
  --dual_branch_train_mode joint \
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
