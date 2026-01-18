#!/usr/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export BACKBONE=blt
export TOKENIZER_NAME=blt
echo $BACKBONE
echo $TOKENIZER_NAME

NNODE=1 NGPU=4 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/test \
  --model.config configs/blt_transformer_340M.json \
  --model.tokenizer_path /yuchang/lsy_jx/.cache/models/gla-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 1 \
  --training.seq_len 10 \
  --training.context_len 1024 \
  --training.varlen \
  --training.gradient_accumulation_steps 2 \
  --training.steps 30720 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset /ssdwork/yuchang/fineweb-edu/sample/100BT \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.num_workers 32S \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --checkpoint.interval 15360 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 1