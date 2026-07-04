#!/bin/bash
# Paired 200-epoch CIFAR100 fine-tuning from the same DeiT-p8 checkpoint.
#
# Default MODE=parallel uses 8 GPUs as two independent 4-GPU jobs:
#   DeiT    -> GPUs 0,1,2,3
#   MergeNet-> GPUs 4,5,6,7
#
# This answers a transfer question, not a scratch-training question:
#   same pretrained checkpoint + matched common training hyperparams + 200e.

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

MODE="${MODE:-parallel}"  # parallel | sequential | deit | mergenet
DEIT_GPUS="${DEIT_GPUS:-0,1,2,3}"
MN_GPUS="${MN_GPUS:-4,5,6,7}"
DEIT_NPROC="${DEIT_NPROC:-$(count_gpus "${DEIT_GPUS}")}"
MN_NPROC="${MN_NPROC:-$(count_gpus "${MN_GPUS}")}"
DEIT_MASTER_PORT="${DEIT_MASTER_PORT:-29511}"
MN_MASTER_PORT="${MN_MASTER_PORT:-29521}"

DATA_DIR="${DATA_DIR:-/liziqing/yukai/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./work_dirs/classification}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_DIR}/_pair_logs/loadckpt_ft200_${RUN_TAG}}"
DRY_RUN="${DRY_RUN:-0}"
DEBUG_SUBSET="${DEBUG_SUBSET:-0}"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-/liziqing/yukai/OpenToMe/work_dirs/classification/cifar100_deit_small_2000e_b200_p8_nofinalcool_minlr0.1/model_best.pth.tar}"
if [[ "${DRY_RUN}" != "1" && ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "[FATAL] PRETRAIN_CKPT not found: ${PRETRAIN_CKPT}" >&2
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
RESUME="${RESUME:-auto}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"

MN_LOCAL_DEPTH="${MN_LOCAL_DEPTH:-2}"
MN_LATENT_DEPTH="${MN_LATENT_DEPTH:-10}"
MN_LAMBDA_LOCAL="${MN_LAMBDA_LOCAL:-4.0}"
MN_SOFT_TOPK="${MN_SOFT_TOPK:-1}"
MN_SOFT_TOPK_AUX_WEIGHT="${MN_SOFT_TOPK_AUX_WEIGHT:-0.10}"

if [[ "${DEIT_NPROC}" -ne 4 || "${MN_NPROC}" -ne 4 ]]; then
  echo "[FATAL] This script is tuned for 4 GPUs per job. Got DEIT_NPROC=${DEIT_NPROC}, MN_NPROC=${MN_NPROC}" >&2
  exit 2
fi
if (( GLOBAL_BATCH % DEIT_NPROC != 0 || GLOBAL_BATCH % MN_NPROC != 0 )); then
  echo "[FATAL] GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by both 4-GPU jobs" >&2
  exit 2
fi
DEIT_BATCH_SIZE="${DEIT_BATCH_SIZE:-$((GLOBAL_BATCH / DEIT_NPROC))}"
MN_BATCH_SIZE="${MN_BATCH_SIZE:-$((GLOBAL_BATCH / MN_NPROC))}"
if [[ $((DEIT_BATCH_SIZE * DEIT_NPROC)) -ne "${GLOBAL_BATCH}" ]]; then
  echo "[FATAL] DeiT effective global batch mismatch: ${DEIT_BATCH_SIZE} * ${DEIT_NPROC} != ${GLOBAL_BATCH}" >&2
  exit 2
fi
if [[ $((MN_BATCH_SIZE * MN_NPROC)) -ne "${GLOBAL_BATCH}" ]]; then
  echo "[FATAL] MergeNet effective global batch mismatch: ${MN_BATCH_SIZE} * ${MN_NPROC} != ${GLOBAL_BATCH}" >&2
  exit 2
fi

DEIT_EXP="${DEIT_EXP:-cifar100_deit_loadckpt_ft${EPOCHS}_4gpu_p${PATCH_SIZE}_b${GLOBAL_BATCH}_lr${LR}_${RUN_TAG}}"
MN_EXP="${MN_EXP:-cifar100_mn_loadckpt_ft${EPOCHS}_4gpu_p${PATCH_SIZE}_ld${MN_LOCAL_DEPTH}_lat${MN_LATENT_DEPTH}_lam${MN_LAMBDA_LOCAL}_b${GLOBAL_BATCH}_lr${LR}_${RUN_TAG}}"

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

build_deit_cmd() {
  DEIT_CMD=(
    "${TORCHRUN_BIN}" --nnodes 1 --nproc_per_node "${DEIT_NPROC}"
    --master_addr 127.0.0.1 --master_port "${DEIT_MASTER_PORT}"
    "${SCRIPT_DIR}/../in1k_trainer.py"
    --model deit_small_patch16_224
    --drop_path_rate "${DROP_PATH_RATE}"
    --batch_size "${DEIT_BATCH_SIZE}"
    --initial_checkpoint "${PRETRAIN_CKPT}"
    --experiment "${DEIT_EXP}"
    "${COMMON_ARGS[@]}"
  )
  local resume_ckpt="${OUTPUT_DIR}/${DEIT_EXP}/last.pth.tar"
  if [[ "${RESUME}" == "auto" && -f "${resume_ckpt}" ]]; then
    DEIT_CMD=("${DEIT_CMD[@]/--initial_checkpoint/${DUMMY_INITIAL_CHECKPOINT_MARKER:-__REMOVE_INITIAL_CKPT__}}")
    DEIT_CMD+=(--resume "${resume_ckpt}")
  fi
}

build_mn_cmd() {
  MN_CMD=(
    "${TORCHRUN_BIN}" --nnodes 1 --nproc_per_node "${MN_NPROC}"
    --master_addr 127.0.0.1 --master_port "${MN_MASTER_PORT}"
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
    --batch_size "${MN_BATCH_SIZE}"
    --initial_checkpoint "${PRETRAIN_CKPT}"
    --find_unused_parameters false
    --experiment "${MN_EXP}"
    "${COMMON_ARGS[@]}"
  )
  if [[ "${MN_SOFT_TOPK}" == "1" ]]; then
    MN_CMD+=(--soft_topk --soft_topk_aux_weight "${MN_SOFT_TOPK_AUX_WEIGHT}")
  fi
  local resume_ckpt="${OUTPUT_DIR}/${MN_EXP}/last.pth.tar"
  if [[ "${RESUME}" == "auto" && -f "${resume_ckpt}" ]]; then
    MN_CMD=("${MN_CMD[@]/--initial_checkpoint/${DUMMY_INITIAL_CHECKPOINT_MARKER:-__REMOVE_INITIAL_CKPT__}}")
    MN_CMD+=(--resume "${resume_ckpt}")
  fi
}

strip_dummy_initial_checkpoint_marker() {
  local -n arr_ref="$1"
  local cleaned=()
  local skip_next=0
  for item in "${arr_ref[@]}"; do
    if [[ "${skip_next}" == "1" ]]; then
      skip_next=0
      continue
    fi
    if [[ "${item}" == "__REMOVE_INITIAL_CKPT__" ]]; then
      skip_next=1
      continue
    fi
    cleaned+=("${item}")
  done
  arr_ref=("${cleaned[@]}")
}

build_deit_cmd
build_mn_cmd
strip_dummy_initial_checkpoint_marker DEIT_CMD
strip_dummy_initial_checkpoint_marker MN_CMD

mkdir -p "${LOG_ROOT}"

print_config() {
  cat <<EOF
[loadckpt ft200 pair]
  mode          : ${MODE}
  checkpoint    : ${PRETRAIN_CKPT}
  DeiT GPUs     : ${DEIT_GPUS} (nproc=${DEIT_NPROC}, per_gpu=${DEIT_BATCH_SIZE}, global=${GLOBAL_BATCH}, port=${DEIT_MASTER_PORT})
  MergeNet GPUs : ${MN_GPUS} (nproc=${MN_NPROC}, per_gpu=${MN_BATCH_SIZE}, global=${GLOBAL_BATCH}, port=${MN_MASTER_PORT})
  common train  : epochs=${EPOCHS}, lr=${LR}, warmup=${WARMUP_EPOCHS}@${WARMUP_LR}, min_lr_ratio=${MIN_LR_RATIO}, wd=${WEIGHT_DECAY}, drop_path=${DROP_PATH_RATE}
  common aug    : mixup=${MIXUP}, cutmix=${CUTMIX}, mixup_mode=${MIXUP_MODE}, smoothing=${SMOOTHING}, aa=${AA}, reprob=${REPROB}
  ema           : ${MODEL_EMA} decay=${MODEL_EMA_DECAY}
  debug_subset  : ${DEBUG_SUBSET}
  MergeNet only : ld=${MN_LOCAL_DEPTH}, lat=${MN_LATENT_DEPTH}, lambda=${MN_LAMBDA_LOCAL}, soft_topk=${MN_SOFT_TOPK}, aux=${MN_SOFT_TOPK_AUX_WEIGHT}
  DeiT exp      : ${DEIT_EXP}
  MergeNet exp  : ${MN_EXP}
  logs          : ${LOG_ROOT}
EOF
}

run_deit() {
  echo "[launch] DeiT -> ${DEIT_GPUS}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' CUDA_VISIBLE_DEVICES="${DEIT_GPUS}" "${DEIT_CMD[@]}"
    printf '\n'
  else
    CUDA_VISIBLE_DEVICES="${DEIT_GPUS}" "${DEIT_CMD[@]}"
  fi
}

run_mn() {
  echo "[launch] MergeNet -> ${MN_GPUS}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' CUDA_VISIBLE_DEVICES="${MN_GPUS}" "${MN_CMD[@]}"
    printf '\n'
  else
    CUDA_VISIBLE_DEVICES="${MN_GPUS}" "${MN_CMD[@]}"
  fi
}

print_config

case "${MODE}" in
  deit)
    run_deit
    ;;
  mergenet|mn)
    run_mn
    ;;
  sequential)
    run_deit
    run_mn
    ;;
  parallel)
    if [[ "${DRY_RUN}" == "1" ]]; then
      run_deit
      run_mn
    else
      echo "[parallel] writing logs to ${LOG_ROOT}/deit.log and ${LOG_ROOT}/mergenet.log"
      CUDA_VISIBLE_DEVICES="${DEIT_GPUS}" "${DEIT_CMD[@]}" > "${LOG_ROOT}/deit.log" 2>&1 &
      pid_deit=$!
      CUDA_VISIBLE_DEVICES="${MN_GPUS}" "${MN_CMD[@]}" > "${LOG_ROOT}/mergenet.log" 2>&1 &
      pid_mn=$!
      status=0
      wait "${pid_deit}" || status=$?
      wait "${pid_mn}" || status=$?
      if [[ "${status}" -ne 0 ]]; then
        echo "[FATAL] one paired job failed; inspect ${LOG_ROOT}" >&2
        exit "${status}"
      fi
      echo "[done] paired jobs finished"
    fi
    ;;
  *)
    echo "[FATAL] MODE must be parallel|sequential|deit|mergenet, got ${MODE}" >&2
    exit 2
    ;;
esac
