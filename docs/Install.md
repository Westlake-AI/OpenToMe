# OpenToMe

### Install

```bash
git clone https://github.com/Westlake-AI/OpenToMe.git
conda create -n opentome python=3.10.0
conda activate opentome
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Developing & Evaluation

1. **Install from `docs/requirements.txt`**

    这个是我当前环境内所有的安装包，可以用来对齐，你也可以直接`pip install -r requirements.txt`来安装环境，但是我个人不推荐，因为可能存在安装包之间顺序的问题。

2. **Install from scoure**
    - **FLA** 先按照上面bash创建base evns，然后记得git clone一下flash-linear-attention，然后按照FLA的需求安装环境 [**flash-linear-attention README.md**](https://github.com/fla-org/flash-linear-attention/blob/main/README.md).

    - **flame** 也是按照它里面的README.md来更新: `pip install -e .`（这个我有点记不清需要需要了，你可以先不用再安装这个试试报不报错）

    - **lmms-evaluation-harness** 这个也是一样先`git clone https://github.com/EleutherAI/lm-evaluation-harness.git`，然后按照README.md安装即可: `pip install -e .`，如果需要eval LongBench的话，需要安装特定的包: `pip install lm_eval['longbench']`

    - [Important] Install specific version of torchtitan: `pip install git+https://github.com/pytorch/torchtitan.git@0b44d4c`

    - **Flash Attention** 这个是需要安装特定版本的，直接复制粘贴即可: `pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`

---

### Training
Here is an example of training with flash linear attention by [**flame**](https://github.com/fla-org/flame)
```bash
#!/usr/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

NNODE=1 NGPU=8 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder RESULTS/PATH \
  --model.config configs/gla_340M.json \
  --model.tokenizer_path TOKENIZER/PATH \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 32 \
  --training.seq_len 2048 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 20480 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset DATASET/PATH \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.streaming \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --training.tensor_parallel_degree 1 \
  --training.disable_loss_parallel \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 1
```

### Evaluation of PPL and Common-sense Resaoning / QA
The evaluation we follow up with the [**flash-linear-attention**](https://github.com/fla-org/flash-linear-attention/blob/main/README.md). Please confirm that the requirements for [**lmms-eval-harness**](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md?plain=1) are satisfied.

Here is an example of PPL evaluation by GLA model.
```bash
#!/usr/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
MODEL='MODEL/PATH'

python -m harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks wikitext \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --show_config
```

### Important [🌟🌟🌟🌟] 四颗星

因为`transformers`不支持flash-attention-linear的Configs，所以我们这边是直接导入os然后手动替换的。所以你这边在跑的时候要注意bash脚本内需要导入：
```
export BACKBONE=MODEL_NAME
echo $BACKBON
```

For example:
```bash
#!/usr/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export BACKBONE=delta_net_340M
echo $BACKBONE
NNODE=1 NGPU=8 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/delta_net_340M_10B/batch1.seqlen32768.grad_acc2.warmup1024.update1.steps20480.lr4e-4 \ # 保存路径 \
  --model.config configs/delta_net_340M.json \  # config文件 \
  --model.tokenizer_path /yuchang/lsy_jx/.cache/models/delta_net-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 1 \
  --training.seq_len 32768 \
  --training.context_len 4096 \
  --training.varlen \
  --training.gradient_accumulation_steps 2 \
  --training.steps 30720 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset /ssdwork/yuchang/fineweb-edu/sample/100BT \  # 数据集路径 \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --checkpoint.interval 15360 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 1
```
Evaluation的时候也是一样，这边我应该都帮你写好了的，问题不大。

### Support Models/Tokenizer 
- ✅ Transformer++
- ✅ GLA
- ✅ DeltaNet
- ✅ Gated-DeltaNet
- ✅ BLT (byte-level)
- ❌ Qwen3-NeXt
- ✅ LLaMA-based Tokenizer
- ✅ Byte-level Tokenizer

说到Tokenizer的不同，只需要`export TOKENIZER_NAME=blt`即可

### Setups [🌟🌟🌟🌟🌟] 五颗星

- **Model Size: 350M** & **1.5B**
    - **340M** 
        1. Total training tokens: ~15B 
        2. batch size: ~0.5M
        3. warmup: ~0.5B
    - **1.5B**
        1. Total training tokens: ~100B
        2. batch size: ~2M 
        2. warmup: ~1B

- **AdamW, LR, wd, grad clip, cosine scheduler, LR_max** 按照我350M的来就行，也就是说你跑1.5B的时候需要修改的只有configs, save path, warmup_steps, seq_len, gradient_accumulation_steps, steps即可，有个计算公式供你参考：
```bash
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
```

*目前OpenToMe repo中大部分只有350M的bash脚本，但是有一个gated-deltenet-1b.sh可以作为你的参考（gla）那个有问题需注意。如果想确认参数到底是否正确，flame贴心的提供了你训练参数对应tokens的数量，在你运行的时候，他会打印出来，见我飞书的那个图。跑之前一定要注意setup是否正确！如果有什么问题可以在飞书里面@我就行~*

