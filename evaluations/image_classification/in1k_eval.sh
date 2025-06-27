#! /bin/bash

### huggingface
export HF_ENDPOINT=https://hf-mirror.com

tome=$1
merge_num=$2

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
./evaluations/image_classification/in1k_example.py \
--model_name vit_base_patch16_224 \
--tome $tome \
--merge_num $merge_num \
--dataset ./data/ImageNet/val_folder \
--inflect -0.5 \

