#!/usr/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export BACKBONE=gated_deltanet
echo $BACKBONE
MODEL='/yuchang/lsy_jx/.cache/opetome_ckpt/check_ok/gated_deltanet_340M-100B'

python -m harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 32 \
    --num_fewshot 0 \
    --device cuda \
    --show_config
