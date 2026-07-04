#!/bin/bash
# 4-GPU MergeNet fine-tuning with a frozen DeiT teacher.
#
# Purpose:
#   Keep the same student initialization and common hyperparameters as the
#   loadckpt FT200 DeiT/MergeNet comparison, then add only logit distillation
#   from the DeiT checkpoint. This targets the single-branch MergeNet accuracy
#   gap without changing global batch or augmentation policy.

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

count_gpus() {
  local list="$1"
  local IFS=','
  read -r -a arr <<< "${list}"
  echo "${#arr[@]}"
}

GPUS="${GPUS:-4,5,6,7}"
NPROC="${NPROC:-$(count_gpus "${GPUS}")}"
MASTER_PORT="${MASTER_PORT:-29531}"
DATA_DIR="${DATA_DIR:-/liziqing/yukai/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./work_dirs/classification}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
DEBUG_SUBSET="${DEBUG_SUBSET:-0}"
RESUME="${RESUME:-auto}"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-/liziqing/yukai/OpenToMe/work_dirs/classification/cifar100_deit_small_2000e_b200_p8_nofinalcool_minlr0.1/model_best.pth.tar}"
TEACHER_CKPT="${TEACHER_CKPT:-${PRETRAIN_CKPT}}"
TEACHER_MODEL="${TEACHER_MODEL:-deit_small_patch16_224}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.5}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${PRETRAIN_CKPT}" ]] || { echo "[FATAL] PRETRAIN_CKPT not found: ${PRETRAIN_CKPT}" >&2; exit 2; }
  [[ -f "${TEACHER_CKPT}" ]] || { echo "[FATAL] TEACHER_CKPT not found: ${TEACHER_CKPT}" >&2; exit 2; }
fi
if [[ "${NPROC}" -ne 4 ]]; then
  echo "[FATAL] This script is tuned for 4 GPUs. Got NPROC=${NPROC}" >&2
  exit 2
fi

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
MIXUP_OFF_EPOCH="${MIXUP_OFF_EPOCH:-0}"
MODEL_EMA="${MODEL_EMA:-1}"
MODEL_EMA_DECAY="${MODEL_EMA_DECAY:-0.9998}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-42}"
CHECKPOINT_HIST="${CHECKPOINT_HIST:-3}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"

MN_LOCAL_DEPTH="${MN_LOCAL_DEPTH:-2}"
MN_LATENT_DEPTH="${MN_LATENT_DEPTH:-10}"
MN_LAMBDA_LOCAL="${MN_LAMBDA_LOCAL:-4.0}"
MN_SOFT_TOPK="${MN_SOFT_TOPK:-1}"
MN_SOFT_TOPK_AUX_WEIGHT="${MN_SOFT_TOPK_AUX_WEIGHT:-0.10}"

if (( GLOBAL_BATCH % NPROC != 0 )); then
  echo "[FATAL] GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by NPROC=${NPROC}" >&2
  exit 2
fi
BATCH_SIZE="${BATCH_SIZE:-$((GLOBAL_BATCH / NPROC))}"
if [[ $((BATCH_SIZE * NPROC)) -ne "${GLOBAL_BATCH}" ]]; then
  echo "[FATAL] effective global batch mismatch: ${BATCH_SIZE} * ${NPROC} != ${GLOBAL_BATCH}" >&2
  exit 2
fi

DW_TAG=$(echo "${DISTILL_WEIGHT}" | tr '.' 'p')
DT_TAG=$(echo "${DISTILL_TEMPERATURE}" | tr '.' 'p')
EXP="${EXP:-cifar100_mn_loadckpt_ft${EPOCHS}_distill_dw${DW_TAG}_t${DT_TAG}_4gpu_p${PATCH_SIZE}_ld${MN_LOCAL_DEPTH}_lat${MN_LATENT_DEPTH}_lam${MN_LAMBDA_LOCAL}_b${GLOBAL_BATCH}_lr${LR}_${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_DIR}/_pair_logs/mn_distill_ft${EPOCHS}_${RUN_TAG}}"
mkdir -p "${LOG_ROOT}"

COMMON_ARGS=(
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
  --mixup_off_epoch "${MIXUP_OFF_EPOCH}"
  --workers "${WORKERS}"
  --amp
  --checkpoint_hist "${CHECKPOINT_HIST}"
  --output "${OUTPUT_DIR}"
  --seed "${SEED}"
  --log_interval "${LOG_INTERVAL}"
)

if [[ "${MODEL_EMA}" == "1" ]]; then
  COMMON_ARGS+=(--model_ema --model_ema_decay "${MODEL_EMA_DECAY}")
fi
if [[ "${DEBUG_SUBSET}" -gt 0 ]]; then
  COMMON_ARGS+=(--debug_subset "${DEBUG_SUBSET}")
fi

CMD=(
  "${TORCHRUN_BIN}" --nnodes 1 --nproc_per_node "${NPROC}"
  --master_addr 127.0.0.1 --master_port "${MASTER_PORT}"
  "${SCRIPT_DIR}/../in1k_trainer.py"
  --model mergenet_small_cls
  --local_depth "${MN_LOCAL_DEPTH}"
  --latent_depth "${MN_LATENT_DEPTH}"
  --dtem_t 1
  --dtem_feat_dim 64
  --metric_grad_scale 0.1
  --source_trace_mode center
  --lambda_local "${MN_LAMBDA_LOCAL}"
  --total_merge_latent 0
  --dtem_window_size 8
  --use_softkmax
  --swa_size 256
  --local_block_window 16
  --drop_path_rate "${DROP_PATH_RATE}"
  --batch_size "${BATCH_SIZE}"
  --initial_checkpoint "${PRETRAIN_CKPT}"
  --distill_teacher_model "${TEACHER_MODEL}"
  --distill_teacher_checkpoint "${TEACHER_CKPT}"
  --distill_weight "${DISTILL_WEIGHT}"
  --distill_temperature "${DISTILL_TEMPERATURE}"
  --find_unused_parameters false
  --experiment "${EXP}"
  "${COMMON_ARGS[@]}"
)

if [[ "${MN_SOFT_TOPK}" == "1" ]]; then
  CMD+=(--soft_topk --soft_topk_aux_weight "${MN_SOFT_TOPK_AUX_WEIGHT}")
fi

resume_ckpt="${OUTPUT_DIR}/${EXP}/last.pth.tar"
if [[ "${RESUME}" == "auto" && -f "${resume_ckpt}" ]]; then
  cleaned=()
  skip_next=0
  for item in "${CMD[@]}"; do
    if [[ "${skip_next}" == "1" ]]; then
      skip_next=0
      continue
    fi
    if [[ "${item}" == "--initial_checkpoint" ]]; then
      skip_next=1
      continue
    fi
    cleaned+=("${item}")
  done
  CMD=("${cleaned[@]}" --resume "${resume_ckpt}")
fi

cat <<EOF
[mn distill ft200]
  GPUs          : ${GPUS} (nproc=${NPROC}, per_gpu=${BATCH_SIZE}, global=${GLOBAL_BATCH}, port=${MASTER_PORT})
  student init  : ${PRETRAIN_CKPT}
  teacher       : ${TEACHER_MODEL} @ ${TEACHER_CKPT}
  distill       : weight=${DISTILL_WEIGHT}, temperature=${DISTILL_TEMPERATURE}
  common train  : epochs=${EPOCHS}, lr=${LR}, warmup=${WARMUP_EPOCHS}@${WARMUP_LR}, min_lr_ratio=${MIN_LR_RATIO}, wd=${WEIGHT_DECAY}, drop_path=${DROP_PATH_RATE}
  common aug    : mixup=${MIXUP}, cutmix=${CUTMIX}, mixup_mode=${MIXUP_MODE}, smoothing=${SMOOTHING}, aa=${AA}, reprob=${REPROB}
  ema           : ${MODEL_EMA} decay=${MODEL_EMA_DECAY}
  debug_subset  : ${DEBUG_SUBSET}
  MergeNet      : ld=${MN_LOCAL_DEPTH}, lat=${MN_LATENT_DEPTH}, lambda=${MN_LAMBDA_LOCAL}, soft_topk=${MN_SOFT_TOPK}, aux=${MN_SOFT_TOPK_AUX_WEIGHT}
  exp           : ${EXP}
  log           : ${LOG_ROOT}/mergenet_distill.log
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' CUDA_VISIBLE_DEVICES="${GPUS}" "${CMD[@]}"
  printf '\n'
else
  CUDA_VISIBLE_DEVICES="${GPUS}" "${CMD[@]}" 2>&1 | tee "${LOG_ROOT}/mergenet_distill.log"
fi
