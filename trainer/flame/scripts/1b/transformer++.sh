#!/usr/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

export BACKBONE=transformer++_340M
echo $BACKBONE

NNODE=1 NGPU=8 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/transformer/batch1.seqlen32768.grad_acc8.warmup477.update1.steps47684.lr1e-3 \
  --model.config configs/transformer_1B.json \
  --model.tokenizer_path /yuchang/lsy_jx/.cache/models/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-3 \
  --lr_scheduler.decay_type cosine \
  --lr_scheduler.warmup_steps 477 \
  --lr_scheduler.lr_min 0.075 \
  --training.batch_size 1 \
  --training.seq_len 32768 \
  --training.context_len 4096 \
  --training.gradient_accumulation_steps 8 \
  --training.steps 47684 \
  --training.varlen \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset /liziqing/fineweb-edu/sample/100BT \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --checkpoint.interval 20000 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 3 \
  --metrics.log_freq 1 \