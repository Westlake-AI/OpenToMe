mkdir -p logs img results

# export BACKBONE=gated_deltanet_340M
# echo $BACKBONE

MODEL_PATH=$1
TOKENIZER_PATH=$2
CON_LEN_MIN=$3
S_LEN=$4
PRE_LEN=$5
SUFFIX=$6

# model_provider=$5
# --model_provider $model_provider \

python -u needle_in_haystack.py --s_len $S_LEN \
    --e_len $PRE_LEN \
    --context_lengths_min $CON_LEN_MIN \
    --context_lengths_max $PRE_LEN \
    --model_path $MODEL_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --model_name_suffix $SUFFIX \
    --simulation_length 0 \
    --context_lengths_num_intervals 13 \
    --document_depth_percent_intervals 10 \
    --sink_size 64 \
    --recent_size 256 \
    --prefilling_chunk_size 32000


python visualize.py \
    --folder_path "results/${MODEL_PATH}_${SUFFIX}/" \
    --model_name "${MODEL_PATH}" \
    --pretrained_len $PRE_LEN
