#!/usr/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

MODEL='/yuchang/lsy_jx/flash-linear-attention/examples/flame/exp/gated_deltanet_340M-100B'

python -m harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --show_config