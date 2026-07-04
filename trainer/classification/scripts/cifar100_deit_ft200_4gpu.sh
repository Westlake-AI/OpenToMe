#!/bin/bash
# Fair DeiT-p8 FT200 baseline on CIFAR-100 (4 GPUs x 50 = global batch 200).
#
# This is the reference transfer baseline for the single compressed MergeNet-B
# comparison: both sides load the SAME DeiT checkpoint and share every common
# hyperparameter (batch, lr schedule, augmentation, EMA).
#
# Usage:
#   bash cifar100_deit_ft200_4gpu.sh                      # run on GPUS=0,1,2,3
#   DRY_RUN=1 bash cifar100_deit_ft200_4gpu.sh            # print command only
#   DEBUG_SUBSET=64 EPOCHS=1 bash cifar100_deit_ft200_4gpu.sh   # smoke test

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${PROJECT_DIR}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ -z "${TORCHRUN_BIN:-}" ]]; then
  if [[ -x /opt/conda/envs/opentome_yk/bin/torchrun ]]; then
    TORCHRUN_BIN=/opt/conda/envs/opentome_yk/bin/torchrun
  else
    TORCHRUN_BIN=torchrun
  fi
fi

count_gpus() { local IFS=','; read -r -a arr <<< "$1"; echo "${#arr[@]}"; }

GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-$(count_gpus "${GPUS}")}"
MASTER_PORT="${MASTER_PORT:-29611}"
DATA_DIR="${DATA_DIR:-/liziqing/yukai/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./work_dirs/classification}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
DEBUG_SUBSET="${DEBUG_SUBSET:-0}"
RESUME="${RESUME:-auto}"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-/liziqing/yukai/OpenToMe/work_dirs/classification/cifar100_deit_small_2000e_b200_p8_nofinalcool_minlr0.1/model_best.pth.tar}"

IMG_SIZE="${IMG_SIZE:-224}"
PATCH_SIZE="${PATCH_SIZE:-8}"
GLOBAL_BATCH="${GLOBAL_BATCH:-200}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.03}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
WARMUP_LR="${WARMUP_LR:-1e-6}"
DROP_PATH_RATE="${DROP_PATH_RATE:-0.10}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
MIXUP="${MIXUP:-0.8}"
CUTMIX="${CUTMIX:-1.0}"
MIXUP_MODE="${MIXUP_MODE:-batch}"
SMOOTHING="${SMOOTHING:-0.1}"
AA="${AA:-rand-m9-mstd0.5-inc1}"
REPROB="${REPROB:-0.25}"
MODEL_EMA="${MODEL_EMA:-1}"
MODEL_EMA_DECAY="${MODEL_EMA_DECAY:-0.9998}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-42}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-3}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"

# --- fairness guards -------------------------------------------------------
if [[ "${DRY_RUN}" != "1" && ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "[FATAL] PRETRAIN_CKPT not found: ${PRETRAIN_CKPT}" >&2; exit 2
fi
if (( GLOBAL_BATCH % NPROC != 0 )); then
  echo "[FATAL] GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by NPROC=${NPROC}" >&2; exit 2
fi
BATCH_SIZE="${BATCH_SIZE:-$((GLOBAL_BATCH / NPROC))}"
if [[ $((BATCH_SIZE * NPROC)) -ne "${GLOBAL_BATCH}" ]]; then
  echo "[FATAL] BATCH_SIZE * NPROC = $((BATCH_SIZE * NPROC)) != GLOBAL_BATCH=${GLOBAL_BATCH}" >&2; exit 2
fi
if [[ "${MIXUP_MODE}" == "batch" && $((BATCH_SIZE % 2)) -ne 0 ]]; then
  echo "[WARN] per-rank batch ${BATCH_SIZE} is odd; trainer pairs the tail sample across ranks to keep mixup fair." >&2
fi

EXP="${EXP:-cifar100_deit_ft${EPOCHS}_${NPROC}gpu_p${PATCH_SIZE}_b${GLOBAL_BATCH}_lr${LR}_${RUN_TAG}}"

CMD=(
  "${TORCHRUN_BIN}" --nnodes 1 --nproc_per_node "${NPROC}"
  --master_addr 127.0.0.1 --master_port "${MASTER_PORT}"
  "${SCRIPT_DIR}/../in1k_trainer.py"
  --model deit_small_patch16_224
  --drop_path_rate "${DROP_PATH_RATE}"
  --batch_size "${BATCH_SIZE}"
  --initial_checkpoint "${PRETRAIN_CKPT}"
  --experiment "${EXP}"
  --data_dir "${DATA_DIR}"
  --dataset CIFAR100
  --train_split train
  --val_split val
  --num_classes 100
  --img_size "${IMG_SIZE}"
  --patch_size "${PATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --sched cosine
  --cooldown_epochs 0
  --min_lr_ratio "${MIN_LR_RATIO}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --warmup_lr "${WARMUP_LR}"
  --clip_grad "${CLIP_GRAD}"
  --mixup "${MIXUP}"
  --cutmix "${CUTMIX}"
  --mixup_mode "${MIXUP_MODE}"
  --smoothing "${SMOOTHING}"
  --aa "${AA}"
  --reprob "${REPROB}"
  --workers "${WORKERS}"
  --amp
  --checkpoint_hist "${CHECKPOINT_HIST}"
  --output "${OUTPUT_DIR}"
  --seed "${SEED}"
  --log_interval "${LOG_INTERVAL}"
)
if [[ "${MODEL_EMA}" == "1" ]]; then
  CMD+=(--model_ema --model_ema_decay "${MODEL_EMA_DECAY}")
fi
if [[ "${DEBUG_SUBSET}" -gt 0 ]]; then
  CMD+=(--debug_subset "${DEBUG_SUBSET}")
fi

resume_ckpt="${OUTPUT_DIR}/${EXP}/last.pth.tar"
if [[ "${RESUME}" == "auto" && -f "${resume_ckpt}" ]]; then
  cleaned=(); skip_next=0
  for item in "${CMD[@]}"; do
    if [[ "${skip_next}" == "1" ]]; then skip_next=0; continue; fi
    if [[ "${item}" == "--initial_checkpoint" ]]; then skip_next=1; continue; fi
    cleaned+=("${item}")
  done
  CMD=("${cleaned[@]}" --resume "${resume_ckpt}")
fi

cat <<EOF
[deit ft baseline]
  GPUs        : ${GPUS} (nproc=${NPROC}, per_gpu=${BATCH_SIZE}, global=${GLOBAL_BATCH}, port=${MASTER_PORT})
  init ckpt   : ${PRETRAIN_CKPT}
  train       : epochs=${EPOCHS}, lr=${LR}, warmup=${WARMUP_EPOCHS}@${WARMUP_LR}, min_lr_ratio=${MIN_LR_RATIO}, wd=${WEIGHT_DECAY}, drop_path=${DROP_PATH_RATE}
  aug         : mixup=${MIXUP}, cutmix=${CUTMIX}, mode=${MIXUP_MODE}, smoothing=${SMOOTHING}, aa=${AA}, reprob=${REPROB}
  ema         : ${MODEL_EMA} decay=${MODEL_EMA_DECAY}
  debug_subset: ${DEBUG_SUBSET}
  exp         : ${EXP}
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' CUDA_VISIBLE_DEVICES="${GPUS}" "${CMD[@]}"; printf '\n'
else
  CUDA_VISIBLE_DEVICES="${GPUS}" "${CMD[@]}"
fi
