#!/bin/bash
# Unified scratch-200e protocol for CIFAR-100 @ 224/p8.
#
# All runs share the SAME protocol (epochs=200, global batch=200, lr=1e-3,
# warmup=20, cosine w/ min_lr_ratio=0.1, mixup/cutmix/randaug/reprob, EMA);
# the ONLY differences are the model and the training method knobs below.
# No run loads any pretrained weights into the student (scratch only).
#
# Usage:
#   MODEL_KIND=deit                          bash cifar100_scratch200.sh
#   MODEL_KIND=mn MN_LOCAL_DEPTH=1 MN_LATENT_DEPTH=11 bash cifar100_scratch200.sh
#   MODEL_KIND=mn KD=1 CURRICULUM=1 SOFT_TOPK=1      bash cifar100_scratch200.sh
#   DRY_RUN=1 ... / DEBUG_SUBSET=64 EPOCHS=1 ... for verification.
#
# KD=1 enables logit + routing + feature distillation from a *frozen teacher*
# (default: DeiT p8 2000e best, 79.01). The student itself stays scratch; the
# teacher only provides soft targets, so this is a training method, not a
# checkpoint transfer.

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

MODEL_KIND="${MODEL_KIND:-deit}"        # deit | mn
KD="${KD:-0}"                           # 1 => logit+routing+feature distillation
CURRICULUM="${CURRICULUM:-0}"           # 1 => lambda ramp LAMBDA_START -> MN_LAMBDA_LOCAL
SOFT_TOPK="${SOFT_TOPK:-0}"             # 1 => soft topk with delayed/ramped aux weight

GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-$(count_gpus "${GPUS}")}"
MASTER_PORT="${MASTER_PORT:-29631}"
DATA_DIR="${DATA_DIR:-/liziqing/yukai/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./work_dirs/classification}"
DRY_RUN="${DRY_RUN:-0}"
DEBUG_SUBSET="${DEBUG_SUBSET:-0}"
RESUME="${RESUME:-auto}"

# --- common protocol (identical for every run in the campaign) ---------------
IMG_SIZE="${IMG_SIZE:-224}"
PATCH_SIZE="${PATCH_SIZE:-8}"
GLOBAL_BATCH="${GLOBAL_BATCH:-200}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
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

# --- MergeNet geometry --------------------------------------------------------
MN_LOCAL_DEPTH="${MN_LOCAL_DEPTH:-1}"
MN_LATENT_DEPTH="${MN_LATENT_DEPTH:-11}"
MN_LAMBDA_LOCAL="${MN_LAMBDA_LOCAL:-4.0}"
MN_DTEM_WINDOW="${MN_DTEM_WINDOW:-8}"
MN_DTEM_FEAT_DIM="${MN_DTEM_FEAT_DIM:-64}"
MN_DTEM_T="${MN_DTEM_T:-1}"
MN_METRIC_GRAD_SCALE="${MN_METRIC_GRAD_SCALE:-0.1}"
MN_SOURCE_TRACE="${MN_SOURCE_TRACE:-center}"
MN_SWA_SIZE="${MN_SWA_SIZE:-256}"
MN_LOCAL_BLOCK_WINDOW="${MN_LOCAL_BLOCK_WINDOW:-16}"

# --- distillation (teacher provides soft targets only) ------------------------
TEACHER_CKPT="${TEACHER_CKPT:-/liziqing/yukai/OpenToMe/work_dirs/classification/cifar100_deit_small_2000e_b200_p8_nofinalcool_minlr0.1/model_best.pth.tar}"
TEACHER_MODEL="${TEACHER_MODEL:-deit_small_patch16_224}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-1.0}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
DISTILL_START_EPOCH="${DISTILL_START_EPOCH:-0}"
DISTILL_RAMP_EPOCHS="${DISTILL_RAMP_EPOCHS:-0}"
ROUTING_WEIGHT="${ROUTING_WEIGHT:-0.5}"
ROUTING_TEMPERATURE="${ROUTING_TEMPERATURE:-1.0}"
ROUTING_START_EPOCH="${ROUTING_START_EPOCH:-0}"
ROUTING_RAMP_EPOCHS="${ROUTING_RAMP_EPOCHS:-10}"
ROUTING_TEACHER_LAYERS="${ROUTING_TEACHER_LAYERS:-9,10,11}"
FEAT_CLS_WEIGHT="${FEAT_CLS_WEIGHT:-1.0}"
FEAT_TOKEN_WEIGHT="${FEAT_TOKEN_WEIGHT:-0.5}"
FEAT_START_EPOCH="${FEAT_START_EPOCH:-0}"
FEAT_RAMP_EPOCHS="${FEAT_RAMP_EPOCHS:-10}"

# --- curriculum / soft-topk schedules -----------------------------------------
LAMBDA_START="${LAMBDA_START-2.0}"
LAMBDA_RAMP_START_EPOCH="${LAMBDA_RAMP_START_EPOCH:-0}"
LAMBDA_RAMP_EPOCHS="${LAMBDA_RAMP_EPOCHS:-50}"
SOFT_TOPK_AUX_WEIGHT="${SOFT_TOPK_AUX_WEIGHT:-0.05}"
SOFT_TOPK_AUX_START_EPOCH="${SOFT_TOPK_AUX_START_EPOCH:-20}"
SOFT_TOPK_AUX_RAMP_EPOCHS="${SOFT_TOPK_AUX_RAMP_EPOCHS:-20}"

# --- guards --------------------------------------------------------------------
if (( GLOBAL_BATCH % NPROC != 0 )); then
  echo "[FATAL] GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by NPROC=${NPROC}" >&2; exit 2
fi
BATCH_SIZE="${BATCH_SIZE:-$((GLOBAL_BATCH / NPROC))}"
if [[ $((BATCH_SIZE * NPROC)) -ne "${GLOBAL_BATCH}" ]]; then
  echo "[FATAL] BATCH_SIZE * NPROC != GLOBAL_BATCH" >&2; exit 2
fi
if [[ "${KD}" == "1" && "${DRY_RUN}" != "1" ]]; then
  [[ -f "${TEACHER_CKPT}" ]] || { echo "[FATAL] TEACHER_CKPT not found: ${TEACHER_CKPT}" >&2; exit 2; }
fi

# --- experiment name -----------------------------------------------------------
if [[ -z "${EXP:-}" ]]; then
  if [[ "${MODEL_KIND}" == "deit" ]]; then
    EXP="c100_scratch${EPOCHS}_deit_p${PATCH_SIZE}_b${GLOBAL_BATCH}"
  else
    EXP="c100_scratch${EPOCHS}_mn_ld${MN_LOCAL_DEPTH}lat${MN_LATENT_DEPTH}_p${PATCH_SIZE}_b${GLOBAL_BATCH}"
    [[ "${KD}" == "1" ]] && EXP="${EXP}_kd"
    [[ "${CURRICULUM}" == "1" ]] && EXP="${EXP}_cur"
    [[ "${SOFT_TOPK}" == "1" ]] && EXP="${EXP}_stk"
  fi
fi

CMD=(
  "${TORCHRUN_BIN}" --nnodes 1 --nproc_per_node "${NPROC}"
  --master_addr 127.0.0.1 --master_port "${MASTER_PORT}"
  "${SCRIPT_DIR}/../in1k_trainer.py"
  --data_dir "${DATA_DIR}"
  --dataset CIFAR100
  --train_split train
  --val_split val
  --num_classes 100
  --img_size "${IMG_SIZE}"
  --patch_size "${PATCH_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --sched cosine
  --cooldown_epochs 0
  --min_lr_ratio "${MIN_LR_RATIO}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --warmup_lr "${WARMUP_LR}"
  --clip_grad "${CLIP_GRAD}"
  --drop_path_rate "${DROP_PATH_RATE}"
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
  --experiment "${EXP}"
  --seed "${SEED}"
  --log_interval "${LOG_INTERVAL}"
)

if [[ "${MODEL_KIND}" == "deit" ]]; then
  CMD+=(--model deit_small_patch16_224)
else
  CMD+=(
    --model mergenet_small_cls
    --local_depth "${MN_LOCAL_DEPTH}"
    --latent_depth "${MN_LATENT_DEPTH}"
    --dtem_t "${MN_DTEM_T}"
    --dtem_feat_dim "${MN_DTEM_FEAT_DIM}"
    --metric_grad_scale "${MN_METRIC_GRAD_SCALE}"
    --source_trace_mode "${MN_SOURCE_TRACE}"
    --lambda_local "${MN_LAMBDA_LOCAL}"
    --total_merge_latent 0
    --dtem_window_size "${MN_DTEM_WINDOW}"
    --use_softkmax
    --swa_size "${MN_SWA_SIZE}"
    --local_block_window "${MN_LOCAL_BLOCK_WINDOW}"
    --find_unused_parameters false
  )
fi

if [[ "${KD}" == "1" ]]; then
  CMD+=(
    --distill_teacher_model "${TEACHER_MODEL}"
    --distill_teacher_checkpoint "${TEACHER_CKPT}"
    --distill_weight "${DISTILL_WEIGHT}"
    --distill_temperature "${DISTILL_TEMPERATURE}"
    --distill_start_epoch "${DISTILL_START_EPOCH}"
    --distill_ramp_epochs "${DISTILL_RAMP_EPOCHS}"
  )
  if [[ "${MODEL_KIND}" == "mn" ]]; then
    CMD+=(
      --routing_distill_weight "${ROUTING_WEIGHT}"
      --routing_distill_temperature "${ROUTING_TEMPERATURE}"
      --routing_distill_start_epoch "${ROUTING_START_EPOCH}"
      --routing_distill_ramp_epochs "${ROUTING_RAMP_EPOCHS}"
      --routing_teacher_layers "${ROUTING_TEACHER_LAYERS}"
      --feat_distill_weight "${FEAT_CLS_WEIGHT}"
      --feat_distill_token_weight "${FEAT_TOKEN_WEIGHT}"
      --feat_distill_start_epoch "${FEAT_START_EPOCH}"
      --feat_distill_ramp_epochs "${FEAT_RAMP_EPOCHS}"
    )
  fi
fi

if [[ "${MODEL_KIND}" == "mn" && "${CURRICULUM}" == "1" && -n "${LAMBDA_START}" ]]; then
  CMD+=(--lambda_start "${LAMBDA_START}"
        --lambda_ramp_start_epoch "${LAMBDA_RAMP_START_EPOCH}"
        --lambda_ramp_epochs "${LAMBDA_RAMP_EPOCHS}")
fi
if [[ "${MODEL_KIND}" == "mn" && "${SOFT_TOPK}" == "1" ]]; then
  CMD+=(--soft_topk
        --soft_topk_aux_weight "${SOFT_TOPK_AUX_WEIGHT}"
        --soft_topk_aux_start_epoch "${SOFT_TOPK_AUX_START_EPOCH}"
        --soft_topk_aux_ramp_epochs "${SOFT_TOPK_AUX_RAMP_EPOCHS}")
fi
if [[ "${MODEL_EMA}" == "1" ]]; then
  CMD+=(--model_ema --model_ema_decay "${MODEL_EMA_DECAY}")
fi
if [[ "${DEBUG_SUBSET}" -gt 0 ]]; then
  CMD+=(--debug_subset "${DEBUG_SUBSET}")
fi

resume_ckpt="${OUTPUT_DIR}/${EXP}/last.pth.tar"
if [[ "${RESUME}" == "auto" && -f "${resume_ckpt}" ]]; then
  CMD+=(--resume "${resume_ckpt}")
fi

cat <<EOF
[scratch200 job]
  exp        : ${EXP}
  model      : ${MODEL_KIND} (mn: ld=${MN_LOCAL_DEPTH} lat=${MN_LATENT_DEPTH} lambda=${MN_LAMBDA_LOCAL})
  method     : KD=${KD} CURRICULUM=${CURRICULUM} SOFT_TOPK=${SOFT_TOPK}
  gpus       : ${GPUS} (nproc=${NPROC}, per_gpu=${BATCH_SIZE}, global=${GLOBAL_BATCH}, port=${MASTER_PORT})
  protocol   : ${EPOCHS}e lr=${LR} warmup=${WARMUP_EPOCHS} min_lr_ratio=${MIN_LR_RATIO} wd=${WEIGHT_DECAY} dp=${DROP_PATH_RATE}
  aug        : mixup=${MIXUP} cutmix=${CUTMIX} aa=${AA} reprob=${REPROB} ema=${MODEL_EMA}@${MODEL_EMA_DECAY}
  teacher    : $([[ "${KD}" == "1" ]] && echo "${TEACHER_MODEL} @ ${TEACHER_CKPT}" || echo none)
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' CUDA_VISIBLE_DEVICES="${GPUS}" "${CMD[@]}"; printf '\n'
else
  CUDA_VISIBLE_DEVICES="${GPUS}" "${CMD[@]}"
fi
