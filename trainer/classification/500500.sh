#!/bin/bash
# Stage2：hybridtomevit_small_cls_dual_ab（joint + fused）warm-start from stage1 branch_a checkpoint.
#
# === 修复演化时间线 ===
# baseline `..._stage2_merge_500e`：
#   epoch 0 eval_top1 ≈ 73 → epoch 20 ≈ 62（−11 点谷底）→ epoch 100 才回到 73。
#   两条根因：
#     (a) AdamW v_t 在 stage2 重新建状态，前 ~4600 步 bias correction 让共享层的
#         有效更新远大于名义 lr；
#     (b) branch_b 私有参数（head_b、fusion_head 中 b 的列）随机初始化，前 ~5000
#         步梯度纯噪声 → 流经共享 encoder 反传污染 stage1 权重。
#
# 修复 v1（requires_grad=False）FAILED：unfreeze 第 1 步 AdamW state.step=0 ⇒
#     update ≈ lr·sign(g) ⇒ NaN。
# 修复 v2（lr_scale=0 单独 param group）FAILED：clip_grad='norm' 把 frozen group
#     的 grad 也算进 global norm，clip_factor ↓ ⇒ head_b/fusion_head 的 effective_lr
#     被压到 < WD 速率 ⇒ 它俩被 WD 拉向 0 ⇒ fused 退化。
# 修复 v3（lr0+ramp + clip-local）部分成功：共享 encoder 已经不再被污染，但
#     eval_top1 epoch 0 仍然从 stage1 的 ~73 跌到 ~60。
#
# === v4（本脚本采用，2026-05-15）：补完两个 warm-start 残余项 ===
# (α) `branch_b.head` 随机初始化导致 `logits_b ≈ 噪声`，`L_b ≈ ln(100) ≈ 4.6` 主导
#     总 loss。即便 lr_scale=0 冻住共享 encoder 的更新，head_b 的 SGD 步会迅速把
#     `|W_b|` 推大，logits_b 开始 confidently 预测错类 → eval 时 50/50 fused 把
#     stage1 的精度直接拉烂。
#     ⇒ 修复：在 `load_branch_a_from_single_model_checkpoint` 里把 stage1 的
#       `head.weight/bias` 同步复制到 `branch_b.head`（两路 head 仍是独立
#       nn.Linear，只是起点完全相同；保持双分支训练协议）。L_b 初始值 ≈ L_a，
#       梯度方向"如何在压缩 cls 上微调"而非"从噪声恢复分类信号"。
#     ⇒ trainer 默认 `--align_branch_b_head_on_load=1`；本脚本显式传 1 以便审计。
# (β) `fusion_head` 默认 0.5·la + 0.5·lb 对 from-scratch 双路同等可信场景合理，
#     warm-start 时 la 是金标准、lb 是噪声，50/50 等于把 a 的精度腰斩。
#     ⇒ 修复：warm-start 时把 fusion_head 重置成 W[i,i]=1.0、W[i,i+nc]=0.0、bias=0，
#       epoch 0 的 logits_fused == logits_a，eval 锁在 stage1 水平；W[i,i+nc]
#       通过 L_fused 的梯度自适应吸收 logits_b。
#     ⇒ trainer 默认 `--fusion_init_on_load=prefer_a`；本脚本显式传以便审计。
#
# 全套依赖（已合入 in1k_trainer.py / mergenet/model.py）：
#   --branch_a_checkpoint                   warm-start 来源
#   --freeze_branch_a_until_epoch / --branch_a_lr_ramp_epochs  + _clip_params_for_step  (v2+v3)
#   --align_branch_b_head_on_load           (v4-α)
#   --fusion_init_on_load                   (v4-β)
#
# 用法：
#   bash 1000s2.sh 2>&1 | tee train_log_stage2_warmstart_1000e_$(date +%Y%m%d_%H%M%S).txt
#
# 可选环境变量：
#   CUDA_VISIBLE_DEVICES=4,5,6,7
#   NPROC_PER_NODE=4
#   DATA_DIR=/path/to/data
#   OUTPUT_DIR=./work_dirs/classification
#   EXP_PREFIX=cifar100_a500e_then_merge500e_p8C
#   STAGE1_EPOCHS=1000                       决定 stage1 ckpt 目录名
#   STAGE1_CKPT=/path/to/model_best.pth.tar  显式 ckpt 路径（覆盖自动查找）
#   STAGE2_EPOCHS=1000
#   FREEZE_BRANCH_A_UNTIL=30                 设 0 关闭 freeze schedule
#   BRANCH_A_LR_RAMP=5
#   ALIGN_HEAD=1                             设 0 复现旧"random branch_b.head"行为
#   FUSION_INIT=prefer_a                     prefer_a | balanced | keep

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR="${DATA_DIR:-/liziqing/yukai/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./work_dirs/classification}"
EXP_PREFIX="${EXP_PREFIX:-cifar100_a500e_then_merge500e_p8C}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-1000}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1000}"
FREEZE_BRANCH_A_UNTIL="${FREEZE_BRANCH_A_UNTIL:-30}"
BRANCH_A_LR_RAMP="${BRANCH_A_LR_RAMP:-5}"
ALIGN_HEAD="${ALIGN_HEAD:-1}"
FUSION_INIT="${FUSION_INIT:-prefer_a}"

STAGE1_EXP="${EXP_PREFIX}_stage1_a_${STAGE1_EPOCHS}e"
STAGE1_CKPT="${STAGE1_CKPT:-${OUTPUT_DIR}/${STAGE1_EXP}/model_best.pth.tar}"

# stage2 实验目录 tag：把所有 warm-start 修复信息编进目录名，避免覆盖旧 run
TAGS=()
if [[ "${FREEZE_BRANCH_A_UNTIL}" -gt 0 ]]; then
  TAGS+=("lr0until${FREEZE_BRANCH_A_UNTIL}_ramp${BRANCH_A_LR_RAMP}")
  TAGS+=("cliplocal")
else
  TAGS+=("nofreeze")
fi
if [[ "${ALIGN_HEAD}" -ne 0 ]]; then
  TAGS+=("headalign")
fi
if [[ "${FUSION_INIT}" != "balanced" ]]; then
  TAGS+=("fusion${FUSION_INIT}")
fi
STAGE2_TAG=$(IFS=_; echo "${TAGS[*]}")
STAGE2_EXP="${EXP_PREFIX}_stage2_merge_${STAGE2_EPOCHS}e_${STAGE2_TAG}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[Stage2][FATAL] stage1 ckpt not found: ${STAGE1_CKPT}"
  echo "  expected dir = ${OUTPUT_DIR}/${STAGE1_EXP}"
  echo "  请先跑 1000s1.sh 或显式 export STAGE1_CKPT=/path/to/model_best.pth.tar"
  exit 2
fi

EXTRA_FREEZE=()
if [[ "${FREEZE_BRANCH_A_UNTIL}" -gt 0 ]]; then
  EXTRA_FREEZE=(
    --freeze_branch_a_until_epoch "${FREEZE_BRANCH_A_UNTIL}"
    --branch_a_lr_ramp_epochs "${BRANCH_A_LR_RAMP}"
  )
  echo "[Stage2] freeze schedule:"
  echo "  - branch_a (含全部共享层) lr_scale=0 直到 epoch ${FREEZE_BRANCH_A_UNTIL}（前向/反向仍走，AdamW v_t 累积）"
  echo "  - 线性 ramp 0→1 共 ${BRANCH_A_LR_RAMP} epochs，从 epoch ${FREEZE_BRANCH_A_UNTIL} 起"
  echo "  - clip_grad='norm' 已在 trainer 内排除 lr_scale=0 的 param group（_clip_params_for_step）"
else
  echo "[Stage2] freeze schedule disabled (FREEZE_BRANCH_A_UNTIL=0)；只靠 head/fusion warm-start"
fi

echo "[Stage2] warm-start from: ${STAGE1_CKPT}"
echo "[Stage2] align_branch_b_head_on_load = ${ALIGN_HEAD}"
echo "[Stage2] fusion_init_on_load        = ${FUSION_INIT}"
echo "[Stage2] dual_ab joint+fused, ${STAGE2_EPOCHS} epochs → ${STAGE2_EXP}"

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
  --branch_a_checkpoint "${STAGE1_CKPT}" \
  --align_branch_b_head_on_load "${ALIGN_HEAD}" \
  --fusion_init_on_load "${FUSION_INIT}" \
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
