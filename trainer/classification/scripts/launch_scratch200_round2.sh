#!/bin/bash
# Round-2 scratch-200e ablations (see docs/MergeNet_Scratch200_Round1_Analysis.md).
#
# Round-1 finding: mn_ld1_kd reached 66.30 (-0.88pp vs DeiT 67.18) with 2.4x lower
# train memory and 1.5x higher eval throughput; gap opened only in the last 50 epochs
# after lambda=4 full compression. Round-2 fixes: faster curriculum, lambda=3 option,
# delayed/heavier routing KD, no soft-topk aux, lighter logit KD.
#
# Usage:
#   bash launch_scratch200_round2.sh              # sequential queue on GPUs 1,3
#   ONLY=v2 bash launch_scratch200_round2.sh      # single job
#   MODE=parallel GPUS=1,3 bash ...               # one job only (default)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LOG_DIR="${PROJECT_DIR}/work_dirs/classification/campaign_logs"
mkdir -p "${LOG_DIR}"

ONLY="${ONLY:-all}"
MODE="${MODE:-sequential}"   # sequential | parallel
GPUS="${GPUS:-1,3}"
MASTER_BASE="${MASTER_BASE:-29641}"

COMMON=(
  MODEL_KIND=mn MN_LOCAL_DEPTH=1 MN_LATENT_DEPTH=11
  KD=1 CURRICULUM=1 SOFT_TOPK=0
  GPUS="${GPUS}"
)

run_job() {
  local name="$1"; shift
  local port="$1"; shift
  local log="${LOG_DIR}/round2_${name}.log"
  echo "[round2] ${name} -> ${log} (GPUs=${GPUS}, port=${port})"
  if [[ "${MODE}" == "parallel" ]]; then
    setsid nohup env "${COMMON[@]}" MASTER_PORT="${port}" "$@" \
      bash "${SCRIPT_DIR}/cifar100_scratch200.sh" >> "${log}" 2>&1 < /dev/null &
    echo "[round2] ${name} pid=$!"
  else
    env "${COMMON[@]}" MASTER_PORT="${port}" "$@" \
      bash "${SCRIPT_DIR}/cifar100_scratch200.sh" >> "${log}" 2>&1
    echo "[round2] ${name} finished"
  fi
}

maybe() {
  local name="$1"; shift
  [[ "${ONLY}" == "all" || "${ONLY}" == "${name}" ]] || return 0
  local port="$1"; shift
  run_job "${name}" "${port}" "$@"
}

# v2 (main): lambda=3, fast curriculum, delayed routing, lighter KD
maybe v2 29641 \
  EXP=c100_scratch200_mn_ld1lat11_kd_v2_lam3 \
  MN_LAMBDA_LOCAL=3.0 LAMBDA_START=2.0 LAMBDA_RAMP_EPOCHS=25 \
  DISTILL_WEIGHT=0.5 ROUTING_WEIGHT=1.0 ROUTING_START_EPOCH=25 ROUTING_RAMP_EPOCHS=5 \
  FEAT_CLS_WEIGHT=0.5 FEAT_TOKEN_WEIGHT=0.25 FEAT_START_EPOCH=25 FEAT_RAMP_EPOCHS=10 \
  ROUTING_AFTER_CURRICULUM=1

# ablation: lambda=4 but faster curriculum + no soft-topk (round-1 had stk)
maybe fastcur25 29642 \
  EXP=c100_scratch200_mn_ld1lat11_kd_fastcur25 \
  MN_LAMBDA_LOCAL=4.0 LAMBDA_START=2.0 LAMBDA_RAMP_EPOCHS=25 \
  DISTILL_WEIGHT=1.0 ROUTING_WEIGHT=0.5 ROUTING_START_EPOCH=0 ROUTING_RAMP_EPOCHS=10 \
  FEAT_CLS_WEIGHT=1.0 FEAT_TOKEN_WEIGHT=0.5 FEAT_START_EPOCH=0 FEAT_RAMP_EPOCHS=10

# ablation: lambda=4, routing only after curriculum with higher weight
maybe rtlate 29643 \
  EXP=c100_scratch200_mn_ld1lat11_kd_rtlate \
  MN_LAMBDA_LOCAL=4.0 LAMBDA_START=2.0 LAMBDA_RAMP_EPOCHS=25 \
  DISTILL_WEIGHT=0.5 ROUTING_WEIGHT=1.0 ROUTING_START_EPOCH=25 ROUTING_RAMP_EPOCHS=5 \
  FEAT_CLS_WEIGHT=0.5 FEAT_TOKEN_WEIGHT=0.25 FEAT_START_EPOCH=25 FEAT_RAMP_EPOCHS=10 \
  ROUTING_AFTER_CURRICULUM=1

echo "[round2] launcher done (mode=${MODE}, only=${ONLY})"
