#!/usr/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

cd /yuchang/lsy_jx/OpenToMe

NNODE=${NNODE:-1}
NGPU=${NGPU:-4}
LOG_RANK=${LOG_RANK:-0}

SEQ_LEN=${SEQ_LEN:-32768}
EPOCHS=${EPOCHS:-5}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-32}
LR=${LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
BETA1=${BETA1:-0.9}
BETA2=${BETA2:-0.999}
EPS=${EPS:-1e-15}
WARMUP_RATIO=${WARMUP_RATIO:-0.01}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-}
DISABLE_FUSED_LOSS=${DISABLE_FUSED_LOSS:-0}
LOG_EVERY=${LOG_EVERY:-1}
SAVE_EVERY=${SAVE_EVERY:-133}
SEED=${SEED:-2222}

RUN_TAG="hg38.seqlen${SEQ_LEN}.epochs${EPOCHS}.global_batch${GLOBAL_BATCH_SIZE}.warmup1pct.${NGPU}gpus.lr${LR}"
# RUN_TAG="hg38.seqlen_schedule0:32k,2:64k,5:128k.epochs${EPOCHS}.global_batch${GLOBAL_BATCH_SIZE}.warmup1pct.${NGPU}gpus.lr${LR}"

MODELS=(
  # "gla_dna_54M_1m:gla:trainer/flame/configs/hg38/gla_dna_54M_1m.json"
  # "hyenadna_small_32k:hyenadna:trainer/flame/configs/hg38/hyenadna_small_32k.json"
  # "delta_net_dna_54M_1m:delta_net:trainer/flame/configs/hg38/delta_net_dna_54M_1m.json"
  # "gated_deltanet_dna_54M_1m:gated_deltanet:trainer/flame/configs/hg38/gated_deltanet_dna_54M_1m.json"
  # "blt_dna_54M_1m:blt:trainer/flame/configs/hg38/blt_transformer_dna_54M_1m.json"
  # "gla_dna_54M_32k:gla:trainer/flame/configs/hg38/gla_dna_54M_32k.json"
  # "hyenadna_small_32k:hyenadna:trainer/flame/configs/hg38/hyenadna_small_32k.json"
  # "delta_net_dna_54M_32k:delta_net:trainer/flame/configs/hg38/delta_net_dna_54M_32k.json"
  # "gated_deltanet_dna_54M_32k:gated_deltanet:trainer/flame/configs/hg38/gated_deltanet_dna_54M_32k.json"
  # "transformer_dna_54M_32k:transformer:trainer/flame/configs/hg38/transformer_dna_54M_32k.json"
  "blt_dna_54M_32k:blt:trainer/flame/configs/hg38/blt_transformer_dna_54M_32k.json"
)

for MODEL_SPEC in "${MODELS[@]}"; do
  IFS=":" read -r MODEL_NAME BACKBONE MODEL_CONFIG <<< "${MODEL_SPEC}"
  OUTPUT_DIR="trainer/flame/exp/dna/${MODEL_NAME}/${RUN_TAG}"

  echo "Training ${MODEL_NAME}"
  echo "  backbone=${BACKBONE}"
  echo "  config=${MODEL_CONFIG}"
  echo "  output_dir=${OUTPUT_DIR}"

  EXTRA_ARGS=()
  if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
    EXTRA_ARGS+=(--attn_implementation "${ATTN_IMPLEMENTATION}")
  fi
  if [[ "${DISABLE_FUSED_LOSS}" == "1" ]]; then
    EXTRA_ARGS+=(--disable_fused_loss)
  fi

  torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NGPU} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0 \
    --local-ranks-filter ${LOG_RANK} \
    --role rank \
    --tee 3 \
    --log-dir "${OUTPUT_DIR}/logs" \
    trainer/flame/flame/train_dna.py \
    --backbone "${BACKBONE}" \
    --model_config "${MODEL_CONFIG}" \
    --fasta data/hg38/hg38.ml.fa.gz \
    --bed data/hg38/human-sequences.bed \
    --split train \
    --output_dir "${OUTPUT_DIR}" \
    --seq_len ${SEQ_LEN} \
    --seq_len_schedule 0:32k,2:64k,5:128k \
    --epochs ${EPOCHS} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --beta1 ${BETA1} \
    --beta2 ${BETA2} \
    --eps ${EPS} \
    --warmup_ratio ${WARMUP_RATIO} \
    --min_lr_ratio ${MIN_LR_RATIO} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --mixed_precision ${MIXED_PRECISION} \
    "${EXTRA_ARGS[@]}" \
    --log_every ${LOG_EVERY} \
    --save_every ${SAVE_EVERY} \
    --seed ${SEED}
done
