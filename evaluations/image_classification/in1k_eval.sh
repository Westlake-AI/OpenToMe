#! /bin/bash

### huggingface
export HF_ENDPOINT=https://hf-mirror.com

cuda=$1
tome=$2
# Support multi round evaulation with different token merge, e.g. 144_98_46_10
merge_num=$3
# /liziqing/ImageNet/val
dataset=$4
gpus=$5
mode=$6
# Model name parameter with default value
model_name=${7:-vit_base_patch16_224}

CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=$gpus \
./evaluations/image_classification/in1k_example.py \
--model_name $model_name \
--tome $tome \
--merge_num $merge_num \
--dataset $dataset \
--inflect -0.5 \
--tracking_mode $mode \

