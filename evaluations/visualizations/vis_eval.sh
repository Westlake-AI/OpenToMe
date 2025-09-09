#! /bin/bash

### huggingface
export HF_ENDPOINT=https://hf-mirror.com

image_path=$1
tome=$2
merge_num=$3
gpu=$4
mode=$5
# Model name parameter with default value
model_name=${6:-vit_base_patch16_224}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$gpu \
evaluations/visualizations/tome_visualization.py \
--image_path $image_path \
--model_name $model_name \
--tome $tome \
--merge_num $merge_num \
--inflect -0.5 \
--save_vis True \
--tracking_mode $mode \

