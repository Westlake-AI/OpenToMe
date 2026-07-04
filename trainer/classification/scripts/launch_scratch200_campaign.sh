#!/bin/bash
# Launch the 4-job scratch-200e campaign on 8 GPUs, detached from the terminal
# (setsid + nohup, so closing the window does not kill training).
#
#   Job A  baseline_deit       GPUs 4,5   pure scratch DeiT-S p8
#   Job B  mn_ld1_plain        GPUs 2,3   MergeNet-B ld1/lat11, architecture only
#   Job C  mn_ld1_kd           GPUs 6,7   ld1/lat11 + KD + curriculum + soft-topk
#   Job D  mn_ld2_kd           GPUs 0,1   ld2/lat10 + KD + curriculum + soft-topk
#
# Usage:
#   bash launch_scratch200_campaign.sh            # launch all four
#   ONLY=deit bash launch_scratch200_campaign.sh  # launch a single job
#   Logs: work_dirs/classification/campaign_logs/<job>.log

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LOG_DIR="${PROJECT_DIR}/work_dirs/classification/campaign_logs"
mkdir -p "${LOG_DIR}"

ONLY="${ONLY:-all}"

launch() {
  local name="$1"; shift
  if [[ "${ONLY}" != "all" && "${ONLY}" != "${name}" ]]; then
    return 0
  fi
  local log="${LOG_DIR}/${name}.log"
  echo "[launch] ${name} -> ${log}"
  setsid nohup env "$@" bash "${SCRIPT_DIR}/cifar100_scratch200.sh" \
    >> "${log}" 2>&1 < /dev/null &
  echo "[launch] ${name} pid=$!"
}

launch deit \
  MODEL_KIND=deit GPUS=4,5 MASTER_PORT=29631

launch mn_ld1_plain \
  MODEL_KIND=mn MN_LOCAL_DEPTH=1 MN_LATENT_DEPTH=11 \
  GPUS=2,3 MASTER_PORT=29632

launch mn_ld1_kd \
  MODEL_KIND=mn MN_LOCAL_DEPTH=1 MN_LATENT_DEPTH=11 \
  KD=1 CURRICULUM=1 SOFT_TOPK=1 \
  GPUS=6,7 MASTER_PORT=29633

launch mn_ld2_kd \
  MODEL_KIND=mn MN_LOCAL_DEPTH=2 MN_LATENT_DEPTH=10 \
  KD=1 CURRICULUM=1 SOFT_TOPK=1 \
  GPUS=0,1 MASTER_PORT=29634

echo "[launch] done. tail -f ${LOG_DIR}/<job>.log to watch."
