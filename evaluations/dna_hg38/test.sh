#!/usr/bin/env bash
set -euo pipefail

ROOT=/yuchang/lsy_jx/OpenToMe
PYTHON=${PYTHON:-/opt/conda/envs/fla/bin/python}
MAX_LENGTH=${MAX_LENGTH:-32768}
DEVICE=${DEVICE:-cuda}
SPLIT=${SPLIT:-test}

mkdir -p outputs

# models=(
#   hyenadna-small-32k-seqlen-hf
#   hyenadna-medium-160k-seqlen-hf
#   hyenadna-medium-450k-seqlen-hf
#   hyenadna-large-1m-seqlen-hf
# )

# for model in "${models[@]}"; do
#   echo "Evaluating ${model} on ${SPLIT}, max_length=${MAX_LENGTH}"
#   "${PYTHON}" hyenadna_hg38_ppl.py \
#     --model_dir "${ROOT}/models/${model}" \
#     --fasta "${ROOT}/data/hg38/hg38.ml.fa.gz" \
#     --bed "${ROOT}/data/hg38/human-sequences.bed" \
#     --split "${SPLIT}" \
#     --max_length "${MAX_LENGTH}" \
#     --device "${DEVICE}" \
#     --output "outputs/${model}_${SPLIT}_ppl_${MAX_LENGTH}.json"
# done

models=(
  gla_dna_54M_32k
  hyenadna_small_32k
  delta_net_dna_54M_32k
  gated_deltanet_dna_54M_32k
  transformer_dna_54M_32k
  # blt_dna_54M_32k
)

# model_tag=hg38.seqlen32768.epochs5.global_batch256.warmup1pct.4gpus.lr1e-3/step-665
model_tag=hg38.seqlen_schedule0:32k,2:64k,5:128k.epochs5.global_batch256.warmup1pct.4gpus.lr1e-3/step-665

for model in "${models[@]}"; do
  echo "Evaluating ${model} on ${SPLIT}, max_length=${MAX_LENGTH}"
  "${PYTHON}" hyenadna_hg38_ppl.py \
    --model_dir /yuchang/lsy_jx/OpenToMe/trainer/flame/exp/dna/${model}/${model_tag} \
    --fasta "${ROOT}/data/hg38/hg38.ml.fa.gz" \
    --bed "${ROOT}/data/hg38/human-sequences.bed" \
    --split "${SPLIT}" \
    --max_length "${MAX_LENGTH}" \
    --device "${DEVICE}" \
    --output "outputs/${model}_${SPLIT}_ppl_${MAX_LENGTH}.json"
done
