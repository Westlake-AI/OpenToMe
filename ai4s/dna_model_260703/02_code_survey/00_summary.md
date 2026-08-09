# 代码调研总览

> 本文件汇总三份深度分析（`01`–`03`）+ 对 OpenToMe 本 checkout 的直接实测结论。
> 面向核心任务：在 OpenToMe 中复现 Transformer++ / DeltaNet / HNet 的 DNA 预训练与评估。
> 整理日期：2026-07-03。

## 三份深度分析文件

| 文件 | 覆盖内容 |
|---|---|
| `01_hyenadna_analysis.md` | HyenaDNA 的 hg38 数据管线、因果 LM 训练逻辑、PPL/下游评估逻辑、复现坑 |
| `02_opentome_flame_fla_analysis.md` | OpenToMe/flame/fla 真实关系、数据格式、模型选择、DNA/HNet 迁移路径 |
| `03_hnet_benchmarks_analysis.md` | HNet 架构与可移植性、GUE/NT-bench 任务与指标、Caduceus↔HyenaDNA 同源 |

## 对 OpenToMe 本 checkout 的直接实测（关键更正）

**上游 vs 本 checkout**：上游 OpenToMe 是 ViT token-merging 工具箱（README 标题证实）；但**本 checkout 已被扩展为 LM 预训练超集**，实测证据：

| 证据 | 路径 | 说明 |
|---|---|---|
| 内置 flame 训练器 | `trainer/flame/` | 含 configs/scripts/train.sh/flame 模块 |
| HF 模型副本 | `opentome/models/{transformer,delta_net,gated_deltanet,gla,gsa,blt,qwen3_next,mergenet_nlp}` | 均可被 flame 训练 |
| MergeNet 已 HF 注册 | `opentome/models/mergenet_nlp/__init__.py` | `AutoModelForCausalLM.register(MergeNetConfig, MergeNetForCausalLM)` |
| byte-level 训练脚本 | `trainer/flame/scripts/byte/{transformer++,deltanet,mergenet}.sh` | `TOKENIZER_NAME=blt`，seq_len 32768 |
| 模型配置 | `trainer/flame/configs/{transformer,delta_net,mergenet}_340M.json` | 340M/1B 级别现成 |
| DNA 内容 | （无） | 全 repo `grep dna/hg38/genome/fasta` = 0 命中（仅 1 处 MMLU 提示词） |

**MergeNet = HNet/MergeDNA 同族**：`mergenet_nlp/model.py` 结构为 `LocalEncoderNLP`（局部）+ `LatentModel`（全局潜在）+ `LocalDecoder`（局部）+ `DTEMBlock`（可微 token 合并）+ 交叉注意力上/下采样，基于 `fla.layers`。配置键 `num_local_layers/num_latent_layers/dtem_window_size/lambda_local` 印证分层合并设计。与 goombalab HNet 的 encoder–main–decoder + 动态分块、MergeDNA 的 Latent Encoder/Decoder + Local Decoder 属同一思想（MergeNet/MergeDNA 均出自 Westlake-AI）。

**flame 数据契约**（`trainer/flame/flame/data.py` 实测）：HF `datasets`，样本读取 `text` 或 `content` 字段并在线 tokenize；支持流式、多数据集 interleave、变长打包（`--training.varlen`）。→ DNA 数据 = 含 `text` 列（核苷酸串）的 HF dataset。

**模型分发**（`trainer/flame/flame/train.py:43-68` 实测）：按 `BACKBONE` 环境变量 `import opentome.models.<x>`；`delta_net`→delta_net，`transformer++`→transformer，`mergenet`→mergenet_nlp。

## 一句话结论

**模型侧（Transformer++/DeltaNet/MergeNet）与训练引擎侧（flame）本 repo 已就绪；数据侧（hg38→HF dataset + DNA tokenizer）与评估侧（GUE/NT-bench/GenomicBenchmarks 接入）是需要新写的部分。** HNet 复现优先用现成 MergeNet，严格版按 flame custom_models 模式包 HF adapter。
