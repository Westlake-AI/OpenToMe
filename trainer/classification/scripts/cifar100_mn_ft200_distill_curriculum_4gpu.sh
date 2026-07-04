#!/bin/bash
# Single compressed MergeNet-B FT200 with the full targeted-supervision stack:
#
#   1. logit distillation from the DeiT teacher (best known single-loss result);
#   2. routing distillation: teacher CLS-attention supervises the student's DTEM
#      token-strength distribution on the same 28x28 patch grid, giving the
#      top-k selection a semantic signal instead of merge-coverage only;
#   3. feature distillation: cosine alignment of the student latent CLS (and
#      optionally the gathered retained tokens) with teacher final features;
#   4. compression curriculum: effective lambda ramps LAMBDA_START -> LAMBDA_LOCAL
#      so the early hard top-k does not lock onto an untrained metric. Best
#      checkpoints are only eligible after the ramp completes (fairness guard);
#   5. delayed/ramped soft_topk aux weight (history: aux=0.3 from epoch 0 hurts).
#
# Fairness: same init checkpoint, same teacher checkpoint, global batch 200,
# and the same common hyperparameters as cifar100_deit_ft200_4gpu.sh.
#
# Usage:
#   bash cifar100_mn_ft200_distill_curriculum_4gpu.sh
#   DRY_RUN=1 bash cifar100_mn_ft200_distill_curriculum_4gpu.sh
#   DEBUG_SUBSET=64 EPOCHS=1 bash cifar100_mn_ft200_distill_curriculum_4gpu.sh

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

GPUS="${GPUS:-4,5,6,7}"
NPROC="${NPROC:-$(count_gpus "${GPUS}")}"
MASTER_PORT="${MASTER_PORT:-29621}"
DATA_DIR="${DATA_DIR:-/liziqing/yukai/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./work_dirs/classification}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
DEBUG_SUBSET="${DEBUG_SUBSET:-0}"
RESUME="${RESUME:-auto}"

# Same checkpoint for student init and teacher => strictly fair transfer setup.
PRETRAIN_CKPT="${PRETRAIN_CKPT:-/liziqing/yukai/OpenToMe/work_dirs/classification/cifar100_deit_small_2000e_b200_p8_nofinalcool_minlr0.1/model_best.pth.tar}"
TEACHER_CKPT="${TEACHER_CKPT:-${PRETRAIN_CKPT}}"
TEACHER_MODEL="${TEACHER_MODEL:-deit_small_patch16_224}"

# --- targeted supervision knobs ---------------------------------------------
DISTILL_WEIGHT="${DISTILL_WEIGHT:-1.0}"          # logit KD; dw=1.0/T=2 is the best known single-loss config (76.36)
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
DISTILL_START_EPOCH="${DISTILL_START_EPOCH:-0}"
DISTILL_RAMP_EPOCHS="${DISTILL_RAMP_EPOCHS:-0}"

ROUTING_WEIGHT="${ROUTING_WEIGHT:-0.5}"          # KL(teacher CLS-attn || student size distribution)
ROUTING_TEMPERATURE="${ROUTING_TEMPERATURE:-1.0}"
ROUTING_START_EPOCH="${ROUTING_START_EPOCH:-0}"
ROUTING_RAMP_EPOCHS="${ROUTING_RAMP_EPOCHS:-10}"
ROUTING_TEACHER_LAYERS="${ROUTING_TEACHER_LAYERS:-9,10,11}"  # avg CLS-attn of the last 3 blocks

FEAT_CLS_WEIGHT="${FEAT_CLS_WEIGHT:-1.0}"        # cosine on final CLS feature
FEAT_TOKEN_WEIGHT="${FEAT_TOKEN_WEIGHT:-0.5}"    # cosine on gathered retained tokens
FEAT_START_EPOCH="${FEAT_START_EPOCH:-0}"
FEAT_RAMP_EPOCHS="${FEAT_RAMP_EPOCHS:-10}"

LAMBDA_START="${LAMBDA_START-2.0}"               # curriculum: keep 2x tokens early...
LAMBDA_RAMP_START_EPOCH="${LAMBDA_RAMP_START_EPOCH:-0}"
LAMBDA_RAMP_EPOCHS="${LAMBDA_RAMP_EPOCHS:-50}"   # ...ramp to LAMBDA_LOCAL by epoch 50
                                                 # set LAMBDA_START="" to disable the curriculum

SOFT_TOPK="${SOFT_TOPK:-1}"
SOFT_TOPK_AUX_WEIGHT="${SOFT_TOPK_AUX_WEIGHT:-0.05}"
SOFT_TOPK_AUX_START_EPOCH="${SOFT_TOPK_AUX_START_EPOCH:-20}"
SOFT_TOPK_AUX_RAMP_EPOCHS="${SOFT_TOPK_AUX_RAMP_EPOCHS:-20}"

# --- MergeNet-B geometry (params ~ DeiT-S: 2 local + 10 latent = 12 blocks) ---
MN_LOCAL_DEPTH="${MN_LOCAL_DEPTH:-2}"
MN_LATENT_DEPTH="${MN_LATENT_DEPTH:-10}"
MN_LAMBDA_LOCAL="${MN_LAMBDA_LOCAL:-4.0}"
MN_DTEM_WINDOW="${MN_DTEM_WINDOW:-8}"
MN_DTEM_FEAT_DIM="${MN_DTEM_FEAT_DIM:-64}"
MN_DTEM_T="${MN_DTEM_T:-1}"
MN_METRIC_GRAD_SCALE="${MN_METRIC_GRAD_SCALE:-0.1}"
MN_SOURCE_TRACE="${MN_SOURCE_TRACE:-center}"
MN_SWA_SIZE="${MN_SWA_SIZE:-256}"
MN_LOCAL_BLOCK_WINDOW="${MN_LOCAL_BLOCK_WINDOW:-16}"

# --- common (must match the DeiT baseline) -----------------------------------
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

# --- fairness guards ---------------------------------------------------------
if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${PRETRAIN_CKPT}" ]] || { echo "[FATAL] PRETRAIN_CKPT not found: ${PRETRAIN_CKPT}" >&2; exit 2; }
  [[ -f "${TEACHER_CKPT}" ]] || { echo "[FATAL] TEACHER_CKPT not found: ${TEACHER_CKPT}" >&2; exit 2; }
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

DW_TAG=$(echo "${DISTILL_WEIGHT}" | tr '.' 'p')
RW_TAG=$(echo "${ROUTING_WEIGHT}" | tr '.' 'p')
FW_TAG=$(echo "${FEAT_CLS_WEIGHT}" | tr '.' 'p')
LS_TAG=$(echo "${LAMBDA_START:-off}" | tr '.' 'p')
EXP="${EXP:-cifar100_mn_ft${EPOCHS}_kd${DW_TAG}_rt${RW_TAG}_ft${FW_TAG}_lam${LS_TAG}to${MN_LAMBDA_LOCAL}_${NPROC}gpu_p${PATCH_SIZE}_ld${MN_LOCAL_DEPTH}_lat${MN_LATENT_DEPTH}_b${GLOBAL_BATCH}_${RUN_TAG}}"

CMD=(
  "${TORCHRUN_BIN}" --nnodes 1 --nproc_per_node "${NPROC}"
  --master_addr 127.0.0.1 --master_port "${MASTER_PORT}"
  "${SCRIPT_DIR}/../in1k_trainer.py"
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
  --drop_path_rate "${DROP_PATH_RATE}"
  --batch_size "${BATCH_SIZE}"
  --initial_checkpoint "${PRETRAIN_CKPT}"
  --distill_teacher_model "${TEACHER_MODEL}"
  --distill_teacher_checkpoint "${TEACHER_CKPT}"
  --distill_weight "${DISTILL_WEIGHT}"
  --distill_temperature "${DISTILL_TEMPERATURE}"
  --distill_start_epoch "${DISTILL_START_EPOCH}"
  --distill_ramp_epochs "${DISTILL_RAMP_EPOCHS}"
  --routing_distill_weight "${ROUTING_WEIGHT}"
  --routing_distill_temperature "${ROUTING_TEMPERATURE}"
  --routing_distill_start_epoch "${ROUTING_START_EPOCH}"
  --routing_distill_ramp_epochs "${ROUTING_RAMP_EPOCHS}"
  --routing_teacher_layers "${ROUTING_TEACHER_LAYERS}"
  --feat_distill_weight "${FEAT_CLS_WEIGHT}"
  --feat_distill_token_weight "${FEAT_TOKEN_WEIGHT}"
  --feat_distill_start_epoch "${FEAT_START_EPOCH}"
  --feat_distill_ramp_epochs "${FEAT_RAMP_EPOCHS}"
  --find_unused_parameters false
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

if [[ -n "${LAMBDA_START}" ]]; then
  CMD+=(--lambda_start "${LAMBDA_START}"
        --lambda_ramp_start_epoch "${LAMBDA_RAMP_START_EPOCH}"
        --lambda_ramp_epochs "${LAMBDA_RAMP_EPOCHS}")
fi
if [[ "${SOFT_TOPK}" == "1" ]]; then
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
  cleaned=(); skip_next=0
  for item in "${CMD[@]}"; do
    if [[ "${skip_next}" == "1" ]]; then skip_next=0; continue; fi
    if [[ "${item}" == "--initial_checkpoint" ]]; then skip_next=1; continue; fi
    cleaned+=("${item}")
  done
  CMD=("${cleaned[@]}" --resume "${resume_ckpt}")
fi

cat <<EOF
[mn ft distill+curriculum]
  GPUs          : ${GPUS} (nproc=${NPROC}, per_gpu=${BATCH_SIZE}, global=${GLOBAL_BATCH}, port=${MASTER_PORT})
  student init  : ${PRETRAIN_CKPT}
  teacher       : ${TEACHER_MODEL} @ ${TEACHER_CKPT}
  logit KD      : w=${DISTILL_WEIGHT} T=${DISTILL_TEMPERATURE} start=${DISTILL_START_EPOCH} ramp=${DISTILL_RAMP_EPOCHS}
  routing KD    : w=${ROUTING_WEIGHT} T=${ROUTING_TEMPERATURE} start=${ROUTING_START_EPOCH} ramp=${ROUTING_RAMP_EPOCHS} layers=${ROUTING_TEACHER_LAYERS}
  feature KD    : cls=${FEAT_CLS_WEIGHT} tok=${FEAT_TOKEN_WEIGHT} start=${FEAT_START_EPOCH} ramp=${FEAT_RAMP_EPOCHS}
  lambda currku : start=${LAMBDA_START:-disabled} ramp=${LAMBDA_RAMP_START_EPOCH}+${LAMBDA_RAMP_EPOCHS}e -> ${MN_LAMBDA_LOCAL}
  soft_topk aux : ${SOFT_TOPK} w=${SOFT_TOPK_AUX_WEIGHT} start=${SOFT_TOPK_AUX_START_EPOCH} ramp=${SOFT_TOPK_AUX_RAMP_EPOCHS}
  common train  : epochs=${EPOCHS}, lr=${LR}, warmup=${WARMUP_EPOCHS}@${WARMUP_LR}, min_lr_ratio=${MIN_LR_RATIO}, wd=${WEIGHT_DECAY}, drop_path=${DROP_PATH_RATE}
  common aug    : mixup=${MIXUP}, cutmix=${CUTMIX}, mode=${MIXUP_MODE}, smoothing=${SMOOTHING}, aa=${AA}, reprob=${REPROB}
  ema           : ${MODEL_EMA} decay=${MODEL_EMA_DECAY}
  MergeNet      : ld=${MN_LOCAL_DEPTH} lat=${MN_LATENT_DEPTH} lambda=${MN_LAMBDA_LOCAL} dtem_w=${MN_DTEM_WINDOW} trace=${MN_SOURCE_TRACE}
  debug_subset  : ${DEBUG_SUBSET}
  exp           : ${EXP}
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' CUDA_VISIBLE_DEVICES="${GPUS}" "${CMD[@]}"; printf '\n'
else
  CUDA_VISIBLE_DEVICES="${GPUS}" "${CMD[@]}"
fi
