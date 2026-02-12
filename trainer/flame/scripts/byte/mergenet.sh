#!/usr/bin/bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export BACKBONE=${BACKBONE:-mergenet_64m}
export TOKENIZER_NAME=${TOKENIZER_NAME:-default}

echo "BACKBONE=${BACKBONE}"
echo "TOKENIZER_NAME=${TOKENIZER_NAME}"

# Overridable runtime args
NNODE=${NNODE:-1}
NGPU=${NGPU:-1}
LOG_RANK=${LOG_RANK:-0}

DUMP=${DUMP:-exp/byte/mergenet_debug}
MODEL_CONFIG=${MODEL_CONFIG:-configs/mergenet_64m.json}
TOKENIZER_PATH=${TOKENIZER_PATH:-gpt2}
# Local SlimPajama directory by default (avoid remote HF access)
DATASET=${DATASET:-/ssdwork/yuchang/SlimPajama-6B}
DATASET_NAME=${DATASET_NAME:-}
DATASET_SPLIT=${DATASET_SPLIT:-train}
# Do NOT use wildcard here unless you pass an explicit comma-separated list.
# train.sh flattens args and shell-glob can expand '*' unexpectedly.
DATA_FILES=${DATA_FILES:-}

STEPS=${STEPS:-200}
WARMUP_STEPS=${WARMUP_STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-1}
SEQ_LEN=${SEQ_LEN:-1024}
CONTEXT_LEN=${CONTEXT_LEN:-1024}
GRAD_ACC=${GRAD_ACC:-1}
LR=${LR:-3e-4}
NUM_WORKERS=${NUM_WORKERS:-4}
PREFETCH=${PREFETCH:-2}

extra_dataset_args=()
if [[ -n "${DATASET_NAME}" ]]; then
  extra_dataset_args+=(--training.dataset_name "${DATASET_NAME}")
fi
if [[ -n "${DATA_FILES}" ]]; then
  extra_dataset_args+=(--training.data_files "${DATA_FILES}")
fi

# Avoid disk quota: DISABLE_WANDB=1 or WANDB_MODE=disabled (wandb writes to ~/.cache)
# Or set DUMP=/ssdwork/yuchang/exp/xxx to put logs on a partition with more quota
extra_args=()
if [[ "${DISABLE_WANDB:-0}" == "1" ]] || [[ "${WANDB_MODE:-}" == "disabled" ]]; then
  extra_args+=(--no-metrics.enable_wandb)
  export WANDB_MODE=disabled  # prevent wandb from patching stderr / writing at import
fi

NNODE=${NNODE} NGPU=${NGPU} LOG_RANK=${LOG_RANK} bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder "${DUMP}" \
  --model.config "${MODEL_CONFIG}" \
  --model.tokenizer_path "${TOKENIZER_PATH}" \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr "${LR}" \
  --lr_scheduler.warmup_steps "${WARMUP_STEPS}" \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size "${BATCH_SIZE}" \
  --training.seq_len "${SEQ_LEN}" \
  --training.context_len "${CONTEXT_LEN}" \
  --training.gradient_accumulation_steps "${GRAD_ACC}" \
  --training.steps "${STEPS}" \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset "${DATASET}" \
  --training.dataset_split "${DATASET_SPLIT}" \
  "${extra_dataset_args[@]}" \
  --training.num_workers "${NUM_WORKERS}" \
  --training.prefetch_factor "${PREFETCH}" \
  --training.seed 42 \
  --checkpoint.interval 300 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 1 \
  "${extra_args[@]}"
