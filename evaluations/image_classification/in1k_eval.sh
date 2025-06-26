#! /bin/bash

### huggingface
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
./evaluations/image_classification/in1k_example.py \
--merge_num 100 --dataset ./data/ImageNet/val_folder

