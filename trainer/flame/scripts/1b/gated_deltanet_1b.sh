#!/usr/bin/bash

# ==========================================
# 训练超参数计算说明 (100B 目标)
# ==========================================
# 1. 单步 Token 数 (Total Batch Size):
#    1 (BS) * 32768 (SeqLen) * 4 (GPU) * 16 (GA) = 2,097,152 (2M Tokens)
# 2. Warmup 步数 (1B 目标):
#    1,000,000,000 / 2,097,152 ≈ 477 Steps
# 3. 总步数 (100B 目标):
#    100,000,000,000 / 2,097,152 ≈ 47,684 Steps
# ==========================================
export HF_ENDPOINT=https://hf-mirror.com
export BACKBONE=gated_deltanet_1B
echo $BACKBONE

NNODE=1 NGPU=8 LOG_RANK=0 bash train.sh  \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/gated_deltanet_1B-100B/batch1.seqlen32768.grad_acc8.warmup477.update1.steps47684.lr4e-4 \
  --model.config configs/gated_deltanet_1B.json \
  --model.tokenizer_path /yuchang/lsy_jx/.cache/models/delta_net-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 4e-4 \
  --lr_scheduler.decay_type cosine \
  --lr_scheduler.warmup_steps 477 \
  --lr_scheduler.lr_min 0.1 \
  --training.batch_size 1 \
  --training.seq_len 32768 \
  --training.context_len 4096 \
  --training.gradient_accumulation_steps 8 \
  --training.steps 47684 \
  --training.varlen \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset /ssdwork/yuchang/fineweb-edu/sample/100BT \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --checkpoint.interval 5000 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 3 \
  --metrics.log_freq 1 \