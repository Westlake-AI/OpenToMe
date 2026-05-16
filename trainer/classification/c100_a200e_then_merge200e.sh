#!/bin/bash
# 两阶段训练范式（branch-A 预训 → dual_ab joint+fused）+ stage2 "排异反应" 修复
#
# === 修复演化时间线（2026-05-13） ===
# baseline (`..._stage2_merge_500e`)：
#   epoch 0 eval_top1 = 72.95 → epoch 20 = 61.98（−11 点谷底）→ epoch 100 才回到 73。
#   原因：(a) AdamW v_t 重置后需 ~4600 步稳态，期间分母被低估 → 共享层有效更新远大于名义 lr；
#         (b) branch B 私有参数随机初始化，前 ~5000 步梯度纯噪声 → 污染共享层。
#
# 修复 v1（requires_grad=False，FAILED — NaN at unfreeze）：
#   freeze 期间 AdamW 对 branch_a 跳过更新 → state.step / m_t / v_t = 0；unfreeze 第 1 步
#   bias correction 退化为 update ≈ lr·sign(g) ⇒ 每坐标跳 1e-3 ⇒ 下个 batch forward NaN。
#   实测：epoch 29 = 73.53 → epoch 30 NaN crash。
#
# 修复 v2（lr_scale=0，FAILED — eval 仍从 73 掉到 61）：
#   把 branch_a 切单独 param group，初始 lr_scale=0；保留 requires_grad=True 让 AdamW v_t
#   在 freeze 期间正常累积；unfreeze 时 lr_scale 线性 ramp 0→1。理论上完美，但实测仍崩。
#   根因：clip_grad='norm' 是 GLOBAL L2 范数，branch_a 的 .grad（L_b 流经共享层的大量
#   随机噪声，每元素 ~0.01，共 10M 维，‖g‖₂ ≈ 31）把 total_norm 抬到 31×，clip_factor =
#   1/31 ≈ 0.032 ⇒ head_b / fusion_head 的 effective_lr ≈ 1e-3·0.032 = 3.2e-5 < AdamW
#   WD 速率 5e-5 ⇒ 它俩被 WD 拉向 0 ⇒ fused logits 退化 ⇒ eval 一路掉。
#
# 修复 v3 = v2 + clip-local（本脚本采用）：
#   trainer 新增 _clip_params_for_step()：clip_grad 只把 lr_scale>0 的 group 的参数纳入
#   global norm，frozen 组（branch_a）排除。这样 head_b / fusion_head 的 effective lr 恢复
#   正常，能压过 WD；同时 branch_a 的 grad 仍流过 AdamW，v_t 正常累积，unfreeze 不爆炸。
# 依赖：trainer 实现 --freeze_branch_a_until_epoch / --branch_a_lr_ramp_epochs +
#       _clip_params_for_step（已合入 in1k_trainer.py）。
#
# 用法：
#   bash c100_a200e_then_merge200e.sh 2>&1 | tee train_log_a_then_merge_$(date +%Y%m%d_%H%M%S).txt
#
# 可选环境变量：
#   CUDA_VISIBLE_DEVICES=4,5,6,7         # GPU 选择
#   NPROC_PER_NODE=4                     # 默认 4
#   DATA_DIR=/path/to/data
#   OUTPUT_DIR=./work_dirs/classification
#   EXP_PREFIX=cifar100_a500e_then_merge500e_p8C    # 与已有 stage1 目录前缀对齐
#   STAGE1_EPOCHS=500                    # stage1 epoch 数，决定 STAGE1_EXP 命名与 ckpt 路径
#   STAGE2_EPOCHS=500                    # stage2 epoch 数
#   FREEZE_BRANCH_A_UNTIL=30             # branch_a lr_scale=0 直到此 epoch；设 0 关闭修复（复现旧 baseline）
#   BRANCH_A_LR_RAMP=5                   # unfreeze 后 lr_scale 线性 ramp 0→1 的 epoch 数（设 0 = 硬切换）
#   STAGE1_CKPT=/path/to/model_best.pth.tar          # 显式 ckpt（覆盖自动查找）

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR="${DATA_DIR:-/liziqing/yukai/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./work_dirs/classification}"
EXP_PREFIX="${EXP_PREFIX:-cifar100_a10000e_then_merge500e_p8C}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-1000}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-500}"
FREEZE_BRANCH_A_UNTIL="${FREEZE_BRANCH_A_UNTIL:-30}"
BRANCH_A_LR_RAMP="${BRANCH_A_LR_RAMP:-5}"

STAGE1_EXP="${EXP_PREFIX}_stage1_a_${STAGE1_EPOCHS}e"

# stage2 实验目录加 tag，避免覆盖已有失败 run（baseline / v1 NaN / v2 clip-killed-by-norm）
if [[ "${FREEZE_BRANCH_A_UNTIL}" -gt 0 ]]; then
  STAGE2_TAG="lr0until${FREEZE_BRANCH_A_UNTIL}_ramp${BRANCH_A_LR_RAMP}_cliplocal"
else
  STAGE2_TAG="baseline"
fi
STAGE2_EXP="${EXP_PREFIX}_stage2_merge_${STAGE2_EPOCHS}e_${STAGE2_TAG}"

STAGE1_CKPT="${STAGE1_CKPT:-${OUTPUT_DIR}/${STAGE1_EXP}/model_best.pth.tar}"

mkdir -p "${OUTPUT_DIR}"

# -------------------------
# Stage 1: Branch-A only (skip if ckpt already exists)
# -------------------------
if [[ -f "${STAGE1_CKPT}" ]]; then
  echo "[Stage1] checkpoint exists, skipping stage1: ${STAGE1_CKPT}"
else
  echo "[Stage1] training branch-A only for ${STAGE1_EPOCHS} epochs → ${STAGE1_EXP}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" "${SCRIPT_DIR}/in1k_trainer.py" \
    --data_dir "${DATA_DIR}" \
    --dataset CIFAR100 \
    --train_split train \
    --val_split val \
    --model hybridtomevit_small_cls_branch_a \
    --num_classes 100 \
    --img_size 224 \
    --patch_size 8 \
    --batch_size 50 \
    --epochs "${STAGE1_EPOCHS}" \
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
    --experiment "${STAGE1_EXP}" \
    --seed 42
fi

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[Stage1][FATAL] stage1 ckpt not found after training: ${STAGE1_CKPT}"
  exit 2
fi

# -------------------------
# Stage 2: Dual-branch merge (joint + fused) with freeze schedule
# -------------------------
# EXTRA_FREEZE=()
# if [[ "${FREEZE_BRANCH_A_UNTIL}" -gt 0 ]]; then
#   EXTRA_FREEZE=(
#     --freeze_branch_a_until_epoch "${FREEZE_BRANCH_A_UNTIL}"
#     --branch_a_lr_ramp_epochs "${BRANCH_A_LR_RAMP}"
#   )
#   echo "[Stage2] mitigating rejection reaction (lr_scale schedule):"
#   echo "  - branch_a lr_scale = 0 for epoch 0..${FREEZE_BRANCH_A_UNTIL} (forward/backward run; Adam v_t accumulates)"
#   echo "  - lr_scale linearly ramps 0 → 1 over epoch ${FREEZE_BRANCH_A_UNTIL}..$((FREEZE_BRANCH_A_UNTIL + BRANCH_A_LR_RAMP))"
#   echo "  - lr_scale = 1 from epoch $((FREEZE_BRANCH_A_UNTIL + BRANCH_A_LR_RAMP)) onwards"
# else
#   echo "[Stage2] freeze schedule disabled (FREEZE_BRANCH_A_UNTIL=0); reproducing baseline behaviour."
# fi

# echo "[Stage2] training dual_ab for ${STAGE2_EPOCHS} epochs (merge) → ${STAGE2_EXP}"
# echo "[Stage2] warm-start branch A from: ${STAGE1_CKPT}"

# CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" "${SCRIPT_DIR}/in1k_trainer.py" \
#   --data_dir "${DATA_DIR}" \
#   --dataset CIFAR100 \
#   --train_split train \
#   --val_split val \
#   --model hybridtomevit_small_cls_dual_ab \
#   --num_classes 100 \
#   --img_size 224 \
#   --patch_size 8 \
#   --dtem_t 1 \
#   --dtem_feat_dim 64 \
#   --metric_grad_scale 0.1 \
#   --branch_b_lambda_local 4.0 \
#   --branch_b_total_merge_latent 0 \
#   --branch_b_dtem_window_size 8 \
#   --branch_b_use_softkmax \
#   --branch_b_swa_size 256 \
#   --fusion_type cat_linear \
#   --dual_branch_train_mode joint \
#   --dual_branch_loss_weight 1.0 \
#   --dual_fused_loss_weight 0.2 \
#   --branch_a_checkpoint "${STAGE1_CKPT}" \
#   "${EXTRA_FREEZE[@]}" \
#   --batch_size 50 \
#   --epochs "${STAGE2_EPOCHS}" \
#   --lr 1e-3 \
#   --weight_decay 0.05 \
#   --sched cosine \
#   --clip_grad 1.0 \
#   --warmup_epochs 20 \
#   --mixup 0.8 \
#   --cutmix 1.0 \
#   --smoothing 0.1 \
#   --aa rand-m9-mstd0.5-inc1 \
#   --workers 32 \
#   --amp \
#   --output "${OUTPUT_DIR}" \
#   --experiment "${STAGE2_EXP}" \
#   --seed 42

# echo "[Done] stage1=${OUTPUT_DIR}/${STAGE1_EXP}, stage2=${OUTPUT_DIR}/${STAGE2_EXP}"
