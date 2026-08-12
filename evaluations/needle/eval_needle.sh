#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WORK_DIR="$REPO_ROOT/work_dirs/needle"

MODEL_PATH=${1:?"usage: $0 MODEL_PATH TOKENIZER_PATH CONTEXT_MIN START_LEN END_LEN SUFFIX [METHOD] [extra arguments...]"}
TOKENIZER_PATH=${2:?"tokenizer path is required"}
CON_LEN_MIN=${3:?"minimum context length is required"}
S_LEN=${4:?"start length is required"}
PRE_LEN=${5:?"end length is required"}
SUFFIX=${6:?"run suffix is required"}
METHOD=${7:-none}
if (( $# >= 7 )); then
    shift 7
else
    shift 6
fi

MAX_CAPACITY=512
EXTRA_ARGS=("$@")
for ((index = 0; index < ${#EXTRA_ARGS[@]}; index++)); do
    case "${EXTRA_ARGS[index]}" in
        --max-capacity-prompt|--max_capacity_prompt)
            if (( index + 1 >= ${#EXTRA_ARGS[@]} )); then
                echo "${EXTRA_ARGS[index]} requires a value" >&2
                exit 2
            fi
            MAX_CAPACITY=${EXTRA_ARGS[index + 1]}
            ((index += 1))
            ;;
        --max-capacity-prompt=*|--max_capacity_prompt=*)
            MAX_CAPACITY=${EXTRA_ARGS[index]#*=}
            ;;
    esac
done

PREFILL_ARGS=(--prefilling-chunk-size 32000)
if [[ "$METHOD" != "none" ]]; then
    PREFILL_ARGS=()
fi

python -u "$SCRIPT_DIR/needle_in_haystack.py" \
    --s-len "$S_LEN" \
    --e-len "$PRE_LEN" \
    --context-lengths-min "$CON_LEN_MIN" \
    --context-lengths-max "$PRE_LEN" \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --model-name-suffix "$SUFFIX" \
    --method "$METHOD" \
    --work-dir "$WORK_DIR" \
    --simulation-length 0 \
    --context-lengths-num-intervals 13 \
    --document-depth-percent-intervals 10 \
    --sink-size 64 \
    --window-size 256 \
    "${PREFILL_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

MODEL_NAME=$(basename -- "${MODEL_PATH%/}")
RUN_NAME="${MODEL_NAME}_${SUFFIX}"
if [[ "$METHOD" != "none" ]]; then
    RUN_NAME="${RUN_NAME}_${METHOD}_${MAX_CAPACITY}"
fi
python "$SCRIPT_DIR/visualize.py" \
    --folder-path "$WORK_DIR/results/$RUN_NAME" \
    --output-dir "$WORK_DIR/visualizations" \
    --model-name "$MODEL_NAME" \
    --pretrained-len "$PRE_LEN"
