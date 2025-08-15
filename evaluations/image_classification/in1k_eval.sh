#! /bin/bash
#/zhoujingbo/yk/work/OpenToMe/evaluations/image_classification/in1k_eval.sh
### huggingface
export HF_ENDPOINT=https://hf-mirror.com

cuda=$1 #0
tome=$2 #none tome
# Support multi round evaulation with different token merge, e.g. 144_98_46_10
merge_num=$3 #90
dataset=$4  # /liziqing/ImageNet/val_folder
gpus=$5 #1

CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=$gpus \
./evaluations/image_classification/in1k_example.py \
--model_name vit_base_patch16_224 \
--tome $tome \
--merge_num $merge_num \
--dataset $dataset \
--inflect -0.5 \
# chmod +x /zhoujingbo/yk/work/OpenToMe/evaluations/image_classification/in1k_eval.sh
# /zhoujingbo/yk/work/OpenToMe/evaluations/image_classification/in1k_eval.sh 0 tome 90 /zhoujingbo/lsy/.cache/imagenet/val 1