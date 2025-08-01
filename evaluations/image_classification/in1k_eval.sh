#! /bin/bash

### huggingface
export HF_ENDPOINT=https://hf-mirror.com

cuda=$1
tome=$2
# Support multi round evaulation with different token merge, e.g. 144_98_46_10
merge_num=$3
dataset=$4
gpus=$5

CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=$gpus \
./evaluations/image_classification/in1k_example.py \
--model_name vit_large_patch16_384 \
--tome $tome \
--merge_num $merge_num \
--dataset $dataset \
--inflect -0.5 \

