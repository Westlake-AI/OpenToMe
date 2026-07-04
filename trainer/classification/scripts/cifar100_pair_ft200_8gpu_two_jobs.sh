#!/bin/bash
# Paired fair comparison on 8 GPUs as two independent 4-GPU jobs:
#   DeiT FT200 baseline           -> GPUs 0,1,2,3 (global batch 200 = 4 x 50)
#   MergeNet-B FT200 new strategy -> GPUs 4,5,6,7 (global batch 200 = 4 x 50)
#
# Both jobs load the SAME DeiT checkpoint and share all common hyperparameters,
# so any gap is attributable to the compressed architecture + its supervision.
#
# Usage:
#   bash cifar100_pair_ft200_8gpu_two_jobs.sh
#   DRY_RUN=1 bash cifar100_pair_ft200_8gpu_two_jobs.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

DEIT_GPUS="${DEIT_GPUS:-0,1,2,3}"
MN_GPUS="${MN_GPUS:-4,5,6,7}"
GLOBAL_BATCH="${GLOBAL_BATCH:-200}"
EPOCHS="${EPOCHS:-200}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
DEBUG_SUBSET="${DEBUG_SUBSET:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./work_dirs/classification}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_DIR}/_pair_logs/pair_ft${EPOCHS}_${RUN_TAG}}"

count_gpus() { local IFS=','; read -r -a arr <<< "$1"; echo "${#arr[@]}"; }
DEIT_NPROC=$(count_gpus "${DEIT_GPUS}")
MN_NPROC=$(count_gpus "${MN_GPUS}")

# Explicit fairness validation before launching anything.
for pair in "DeiT:${DEIT_NPROC}" "MergeNet:${MN_NPROC}"; do
  name="${pair%%:*}"; nproc="${pair##*:}"
  if (( GLOBAL_BATCH % nproc != 0 )); then
    echo "[FATAL] ${name}: GLOBAL_BATCH=${GLOBAL_BATCH} not divisible by nproc=${nproc}" >&2; exit 2
  fi
  per_gpu=$((GLOBAL_BATCH / nproc))
  if (( per_gpu * nproc != GLOBAL_BATCH )); then
    echo "[FATAL] ${name}: per_gpu * nproc != GLOBAL_BATCH" >&2; exit 2
  fi
  if (( per_gpu % 2 != 0 )); then
    echo "[WARN] ${name}: per-rank batch ${per_gpu} is odd. batch-mode mixup/cutmix stays fair only" >&2
    echo "       because in1k_trainer pairs the odd tail sample across ranks; verified supported." >&2
  fi
done

mkdir -p "${LOG_ROOT}"

echo "[pair] DeiT   -> GPUs ${DEIT_GPUS} (${DEIT_NPROC} x $((GLOBAL_BATCH / DEIT_NPROC)) = ${GLOBAL_BATCH})"
echo "[pair] MergeNet-> GPUs ${MN_GPUS} (${MN_NPROC} x $((GLOBAL_BATCH / MN_NPROC)) = ${GLOBAL_BATCH})"
echo "[pair] logs   -> ${LOG_ROOT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  GPUS="${DEIT_GPUS}" NPROC="${DEIT_NPROC}" GLOBAL_BATCH="${GLOBAL_BATCH}" EPOCHS="${EPOCHS}" \
    RUN_TAG="${RUN_TAG}" DEBUG_SUBSET="${DEBUG_SUBSET}" DRY_RUN=1 MASTER_PORT="${DEIT_MASTER_PORT:-29611}" \
    bash "${SCRIPT_DIR}/cifar100_deit_ft200_4gpu.sh"
  GPUS="${MN_GPUS}" NPROC="${MN_NPROC}" GLOBAL_BATCH="${GLOBAL_BATCH}" EPOCHS="${EPOCHS}" \
    RUN_TAG="${RUN_TAG}" DEBUG_SUBSET="${DEBUG_SUBSET}" DRY_RUN=1 MASTER_PORT="${MN_MASTER_PORT:-29621}" \
    bash "${SCRIPT_DIR}/cifar100_mn_ft200_distill_curriculum_4gpu.sh"
  exit 0
fi

GPUS="${DEIT_GPUS}" NPROC="${DEIT_NPROC}" GLOBAL_BATCH="${GLOBAL_BATCH}" EPOCHS="${EPOCHS}" \
  RUN_TAG="${RUN_TAG}" DEBUG_SUBSET="${DEBUG_SUBSET}" MASTER_PORT="${DEIT_MASTER_PORT:-29611}" \
  bash "${SCRIPT_DIR}/cifar100_deit_ft200_4gpu.sh" > "${LOG_ROOT}/deit.log" 2>&1 &
pid_deit=$!

GPUS="${MN_GPUS}" NPROC="${MN_NPROC}" GLOBAL_BATCH="${GLOBAL_BATCH}" EPOCHS="${EPOCHS}" \
  RUN_TAG="${RUN_TAG}" DEBUG_SUBSET="${DEBUG_SUBSET}" MASTER_PORT="${MN_MASTER_PORT:-29621}" \
  bash "${SCRIPT_DIR}/cifar100_mn_ft200_distill_curriculum_4gpu.sh" > "${LOG_ROOT}/mergenet.log" 2>&1 &
pid_mn=$!

echo "[pair] launched: deit pid=${pid_deit}, mergenet pid=${pid_mn}"
status=0
wait "${pid_deit}" || status=$?
wait "${pid_mn}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  echo "[FATAL] one paired job failed; inspect ${LOG_ROOT}" >&2
  exit "${status}"
fi
echo "[pair] both jobs finished"
