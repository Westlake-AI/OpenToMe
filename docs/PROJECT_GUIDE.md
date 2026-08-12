# OpenToMe 使用与实现指南

OpenToMe 是一个面向长上下文推理、KV Cache Compression、Attention/Token
Merge、模型实验和评测的研究代码库。本文只保留核心结构和常用启动方式。

## 1. 项目结构

```text
OpenToMe/
├── opentome/
│   ├── compress/              # KV Cache Compression
│   │   ├── base.py            # 配置与 Policy 接口
│   │   ├── cache.py           # CompressedDynamicCache
│   │   └── methods/           # 单个压缩方法
│   ├── models/                # 模型实现与推理适配器
│   │   ├── llama/             # Llama KV compression adapter
│   │   ├── mistral/           # Mistral KV compression adapter
│   │   └── ...                # GLA、GSA、BLT、ToMe 等模型
│   ├── tome/                  # Attention/Token Merge 方法
│   └── timm/                  # ViT/视觉模型 ToMe 方法
├── evaluations/
│   ├── inference/             # KV benchmark、RULER、LongBench
│   ├── needle/                # Needle In A Haystack
│   ├── image_classification/  # ImageNet/ViT 分类评测
│   ├── visualizations/        # ToMe token merge 可视化
│   ├── throughputs/           # 吞吐率和延迟测试
│   ├── dna_hg38/              # DNA 模型 perplexity
│   └── lm_harness/            # lm-evaluation-harness
├── data/                      # 本地评测数据，例如 data/LongBench
├── work_dirs/                 # 所有评测输出（已加入 .gitignore）
└── test/                      # 单元测试和集成测试
```

## 2. 环境准备

推荐使用项目已有的 `fla` 环境：

```bash
conda env create -f fla_environment.yml
conda activate fla
pip install -e .
```

长上下文 GPU 推理建议安装 FlashAttention 2，并使用 `torch.float16` 或
`torch.bfloat16`。Transformers 4.37 和 4.57 的 Llama/Mistral 接口均已兼容。

基础检查：

```bash
python -m py_compile opentome/compress/cache.py \
  opentome/models/llama/kv_compression.py \
  opentome/models/mistral/kv_compression.py
python -m unittest discover -v -s test -p 'test_kv_cache_compression.py'
python -m unittest discover -v -s test -p 'test_kvcache_factory_methods.py'
```

## 3. KV Cache Compression 实现

### 3.1 运行流程

```text
模型加载
  -> patch_model_for_kv_compression(model)
  -> CompressedDynamicCache(config)
  -> prefill: policy.compress_prefill(...)
  -> decode:  policy.update_decode(...)
  -> attention backend: FlashAttention 2 / SDPA / eager
```

`CompressedDynamicCache` 继承 Transformers 的 `DynamicCache`，保存两种长度：

- `logical_length`：真实已经处理的 token 数，用于 RoPE 和位置编码。
- physical length：实际保存在 cache 中的 token 数，用于显存和计算量统计。

新版 Transformers 使用 `layers[*].keys/values`，旧版使用
`key_cache/value_cache`，OpenToMe 的 cache 对两者都提供兼容处理。

### 3.2 已注册方法

| 方法 | 注册名 | 特点 |
| --- | --- | --- |
| StreamingKV / StreamingLLM | `streamingkv`, `streamingllm` | sink token + recent window |
| H2O | `h2o` | 基于历史注意力重要性 |
| SnapKV | `snapkv` | prefill query-aware pooling/selection |
| PyramidKV | `pyramidkv` | 不同层使用不同容量 |
| L2Norm | `l2norm` | key norm 选择 |
| CAM | `cam` | 累积注意力重要性 |
| Quest | `quest` | page/proxy query-aware retrieval，物理 cache 不一定变小 |
| NACL | `nacl` | proxy token + 随机预算 |
| Scissorhands | `scissorhands` | 重要性衰减与 token 选择 |

方法代码位于 `opentome/compress/methods/<method>.py`，统一继承
`KVCompressionPolicy`。新增方法只需：

1. 实现 `compress_prefill`，必要时实现 `compress_decode` 或 `update_decode`。
2. 在 `opentome/compress/methods/__init__.py` 加入 `POLICY_REGISTRY`。
3. 在评测脚本的 `--method` choices 中确认注册名可见。

### 3.3 直接使用示例

```python
from opentome.compress import CompressedDynamicCache, KVCompressionConfig
from opentome.models.kv_compression import patch_model_for_kv_compression

model = ...  # LlamaForCausalLM 或 MistralForCausalLM
patch_model_for_kv_compression(model)
cache = CompressedDynamicCache(KVCompressionConfig(
    method="snapkv",
    max_capacity_prompt=512,
    window_size=32,
    kernel_size=7,
    num_hidden_layers=model.config.num_hidden_layers,
))
outputs = model.generate(
    input_ids,
    past_key_values=cache,
    use_cache=True,
    max_new_tokens=128,
)
```

## 4. Attention/Token Merge 与模型

KV compression 和 ToMe 是不同机制：

- `opentome/compress`：压缩 KV cache，主要影响 decode 阶段的 KV 长度。
- `opentome/tome`：在 attention block 或 token 表示上做 token merge。
- `opentome/timm`：视觉模型/ViT 的 token merge，PiToMe 等方法通常用于
  vision token，不等价于 Llama 文本 token 的 KV compression。
- `opentome/models`：模型 configuration、modeling 和 Transformers 注册/适配。

当前 KV compression adapter 已覆盖 Llama 和 Mistral。其他模型不能直接传入
`patch_model_for_kv_compression`，需要实现对应模型的 attention forward 和 cache
调用契约。

## 5. 评测入口

所有默认输出统一写入 `work_dirs/<task>/`，不会散落在当前工作目录。

### 5.1 LongBench：预测并评分

数据固定读取：`data/LongBench`。

```bash
bash evaluations/inference/longbench/run_longbench.sh \
  /path/to/Llama-3.1-8B-Instruct \
  snapkv \
  qasper \
  2048 \
  flash_attention_2
```

参数顺序：

```text
MODEL_PATH [METHOD] [DATASET] [MAX_CAPACITY] [ATTN_IMPLEMENTATION]
```

结果位置：

```text
work_dirs/longbench/<model>/<method>/longbench/<dataset>.jsonl
work_dirs/longbench/<model>/<method>/longbench/result.json
```

直接评分已有预测：

```bash
python evaluations/inference/longbench/evaluate.py \
  --prediction-path work_dirs/longbench/<model>/snapkv/longbench
```

LongBench-E：

```bash
python evaluations/inference/longbench/evaluate.py \
  --prediction-path work_dirs/longbench/<model>/snapkv/longbench_e \
  --longbench-e
```

### 5.2 KV 性能 benchmark

用于测量吞吐、延迟、峰值显存和物理 cache 大小，不评估答案正确率：

```bash
python -m evaluations.inference.benchmark_kv \
  --model-path /path/to/model \
  --method snapkv \
  --max-capacity-prompt 512 \
  --max-new-tokens 128 \
  --repeat 3
```

默认输出：`work_dirs/benchmark_kv/result.json`。

比较不同方法：

```bash
for method in streamingkv h2o snapkv pyramidkv; do
  python -m evaluations.inference.benchmark_kv \
    --model-path /path/to/model \
    --method "$method" \
    --max-capacity-prompt 512 \
    --output "work_dirs/benchmark_kv/${method}.json"
done
```

### 5.3 RULER

RULER 记录需要包含 `input` 和 `outputs` 字段：

```bash
python -m evaluations.inference.ruler \
  --model-path /path/to/model \
  --method nacl \
  --data-file /path/to/RULER/4096/niah_single_1.jsonl \
  --max-new-tokens 64
```

默认输出：`work_dirs/ruler/predictions.jsonl`。

### 5.4 Needle In A Haystack

```bash
bash evaluations/needle/eval_needle.sh \
  /path/to/model \
  /path/to/tokenizer \
  4096 4096 8192 run01 snapkv
```

输出：

```text
work_dirs/needle/results/
work_dirs/needle/contexts/
work_dirs/needle/visualizations/
```

也可以直接运行：

```bash
python evaluations/needle/needle_in_haystack.py \
  --model-path /path/to/model \
  --tokenizer-path /path/to/tokenizer \
  --method snapkv \
  --s-len 4096 --e-len 8192
```

### 5.5 ImageNet / ViT ToMe

单卡或分布式 ImageNet 评测：

```bash
bash evaluations/image_classification/in1k_eval.sh \
  0 tome 98 /path/to/imagenet/val 1 map vit_base_patch16_224
```

ToMe 可视化：

```bash
bash evaluations/visualizations/vis_eval.sh \
  /path/to/image tome 98 1 matrix vit_base_patch16_224
```

默认结果分别位于 `work_dirs/image_classification` 和
`work_dirs/visualizations`。

### 5.6 Throughput

```bash
python evaluations/throughputs/run_benchmark.py \
  --model-names deit_small_patch16_224 \
  --seq-lens 196 384 \
  --algorithms none tome pitome \
  --target-ratios 0.25 0.5
```

默认输出位于 `work_dirs/throughputs/`。

## 6. 输出目录约定

```text
work_dirs/
├── longbench/
├── benchmark_kv/
├── ruler/
├── needle/
├── image_classification/
├── visualizations/
├── throughputs/
├── dna_hg38/
└── lm_harness/
```

`work_dirs/` 已加入 `.gitignore`。实验结果、日志、预测 JSONL 和可视化图片
都应放在对应任务目录中。

## 7. 常见问题

### 显存不足

优先使用 FlashAttention 2：

```text
--attn-implementation flash_attention_2
```

同时降低 `max-capacity-prompt`、输入长度或 `max-new-tokens`。如果加载的
Transformers/model 不支持 FlashAttention 2，可退回 `sdpa`。

### `LlamaAttention` 没有 `num_heads`

不要直接修改 Transformers 的全局类。使用：

```python
from opentome.models.kv_compression import patch_model_for_kv_compression
patch_model_for_kv_compression(model)
```

适配器已兼容 Transformers 4.37 与 4.57 的 Attention、RoPE、返回值和
DynamicCache 内部结构。

### 方法名不存在

查看注册表：

```python
from opentome.compress import POLICY_REGISTRY
print(sorted(POLICY_REGISTRY))
```

### LongBench 数据集加载失败

优先检查：

```text
data/LongBench/<dataset>.jsonl
```

当前 LongBench runner 默认离线读取该目录；缺失时再根据 `data.py` 的逻辑尝试
Hugging Face 数据源。

