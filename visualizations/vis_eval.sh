#! /bin/bash

### huggingface
export HF_ENDPOINT=https://hf-mirror.com

tome=$1
merge_num=$2

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
visualizations/tome_visualization.py \
--model_name vit_base_patch16_224 \
--tome $tome \
--merge_num $merge_num \
--inflect -0.5 \
--save_vis True \

