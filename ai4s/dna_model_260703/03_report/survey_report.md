# DNA 基础模型复现调研报告

> **核心任务**：在 **OpenToMe**（flash-linear-attention `fla` + `flame` 训练引擎）中，复现已有模型（**Transformer++ / DeltaNet**）与 **HNet** 在 DNA 预训练数据（human hg38）上的**预训练 + 评估**。
> **本报告定位**：文献 + 代码 + 数据 + 基准的系统性调研结论，**不含实现**，用于指导后续复现方案。
> 整理日期：2026-07-03。证据分层：【已核验/有代码】【论文/文档明确】【合理推断】【待确认】。
> 目录：`01_literature/`（28 篇 PDF + INDEX）、`02_code_survey/`（3 份代码分析 + 本 repo 实测）、`03_report/`（本报告）。

---

## 0. TL;DR — 最关键的 5 个结论

1. **OpenToMe 不是你以为的样子，但比你以为的更好。**【已核验】上游 OpenToMe 是 Westlake-AI 的 **ViT token-merging 工具箱**（README 标题即"Toolbox and Benchmark for Token Merging Modules"）。但**当前 checkout 已被扩展为一个语言模型预训练超集**：`trainer/flame/` 内置了一份完整的 flame 训练器，`opentome/models/` 内置了 **Transformer++、DeltaNet、GatedDeltaNet、GLA、GSA、BLT、Qwen3-Next、以及 MergeNet** 的 HF 风格模型副本，`trainer/flame/configs/` 有对应的 340M/1B 配置，`trainer/flame/scripts/byte/` 有 **byte-level 训练脚本**。→ **待复现的 Transformer++ 与 DeltaNet 本 repo 已就绪，可直接训练。**

2. **HNet 无需从零移植 —— 本 repo 已有它的"同族兄弟" MergeNet。**【已核验】`opentome/models/mergenet_nlp/` 是 Westlake 团队自研的**分层 token-merging 语言模型**：`LocalEncoderNLP`（局部编码）+ `LatentModel`（全局潜在层）+ `LocalDecoder`（局部解码）+ `DTEMBlock`（可微 token 合并）+ 交叉注意力上/下采样。这与 **HNet 的 encoder–main–decoder + 动态分块**、以及 **MergeDNA 的 Latent Encoder/Decoder + Local Decoder** 是**同一思想的三个实现**（MergeNet / MergeDNA 均出自 Westlake-AI 同一实验室）。MergeNet 已完成 `AutoModelForCausalLM.register`，可被 flame 直接训练。→ **"复现 HNet" 有两条路径**：(A) 直接用现成的 MergeNet 作为 HNet-spirit 模型；(B) 若要严格复现 goombalab 原版 HNet，则按 flame 的 `custom_models` 模式包一层 HF adapter。

3. **HyenaDNA 是数据与评估管线的权威蓝本，Caduceus 与它同源。**【已核验】HyenaDNA 用 `HG38Dataset`（`src/dataloaders/datasets/hg38_dataset.py`）从 `hg38.ml.fa`(~3GB) + `human-sequences.bed` 采样定长窗口，单核苷酸 `CharacterTokenizer`（vocab=12），因果 LM（next-token）目标，PPL=`exp(mean NLL)`。Caduceus 代码 fork 自 HyenaDNA，配置/数据/评估完全同源。→ **DNA 数据处理与评估逻辑照搬 HyenaDNA 即可，不必另起炉灶。**

4. **三大标准基准的数据与代码来源已定位。**【已核验】**GUE**（28 数据集/7 任务）→ DNABERT-2 repo（Google Drive 下载 + `finetune/train.py`，指标 MCC/F1/acc）；**NT-bench**（18 任务）→ HuggingFace `InstaDeepAI/nucleotide_transformer_downstream_tasks`（指标 MCC/F1/acc）；**Genomic Benchmarks**（8 任务）→ HyenaDNA/Caduceus 内置 dataloader。

5. **OpenToMe 当前 0 行 DNA 代码。**【已核验】全 repo 仅 1 处 `dna` 命中（MMLU 提示词），无 hg38/genome/fasta。→ **数据侧（hg38→HF dataset + DNA tokenizer）与评估侧（三大基准接入）是本项目真正需要新写的部分；模型侧与训练引擎侧已具备。**

---

## 1. 文献调研结论

### 1.1 已下载文献（28 篇，`01_literature/papers/`，详见 `01_literature/INDEX.md`）

以 MergeDNA (arXiv 2511.14806) 的相关工作与参考文献为种子扩充，覆盖：

| 分组 | 论文 | 对复现的意义 |
|---|---|---|
| **代表性开源 DNA 模型** | HyenaDNA★, Caduceus★, DNABERT, DNABERT-2★, Nucleotide Transformer★, GPN, Evo, DNAGPT, HybriDNA, ConvNova | 数据/评估蓝本、基准来源、架构范式对照 |
| **DNA 动态分词**（HNet 的 DNA 对标） | **MergeDNA★**, VQDNA, MxDNA, Omni-DNA | 学习式分词在 DNA 上的落地，直接对应 HNet/MergeNet 思想 |
| **序列骨干与方法** | **HNet★, DeltaNet★, Gated DeltaNet★**, Mamba★, Mamba-2★, Hyena Hierarchy | 待复现模型 + fla 内置骨干 |
| **byte-level 分层建模**（HNet 的 NLP 同族） | BLT, MambaByte, MEGABYTE, SpaceByte, Dynamic Token Pooling | 无分词/动态分块思想谱系，OpenToMe 内置 BLT tokenizer |
| **闭源/标杆** | Enformer | 长程上下文对照 |
| **基准** | Genomic Benchmarks, GenBench（+ GUE→DNABERT-2 / NT-bench→NT） | 评估协议与指标 |

### 1.2 MergeDNA 给出的 DNA 模型全景（种子论文的分类学）

MergeDNA 将 DNA 基础模型按 **4 架构范式 × 4 分词类型** 组织，本调研据此系统扩充：

- **架构**：① SSM（HyenaDNA, Caduceus）② Transformer（DNABERT/-2, NTv1/v2, GROVER, GenSLM）③ Hybrid（Evo, Evo2, HybriDNA）④ CNN（ConvNova）
- **分词**：byte-level（Evo）、k-mer（NTv2）、BPE（DNABERT-2）、**动态分词（VQDNA / MxDNA / MergeDNA）**

**关键洞察**：本项目要复现的 HNet 属于"动态分词/动态分块"路线，其 **DNA 域的直接同类就是 MergeDNA / MxDNA / VQDNA**。这些论文（已下载）应作为"HNet 在 DNA 上表现预期"与"评估设置"的一手参考。

### 1.3 未获取文献（bioRxiv Cloudflare 拦截，需手动补齐）

Evo 2、GROVER、GenSLM、GENA-LM、GPN-MSA —— 见 INDEX.md 末尾获取建议。命令行无法绕过 bioRxiv 人机校验；期刊 OA 版（Nature/PNAS/BMC/PMC）已通过 Europe PMC 成功获取。

---

## 2. 代码调研结论

### 2.1 HyenaDNA repo（数据/训练/评估蓝本）——【已核验】

详见 `02_code_survey/01_hyenadna_analysis.md`。要点：

**数据（DATA）**
- 预训练数据：人类基因组 **hg38**（`hg38.ml.fa` ~3GB，取自 Basenji GCS bucket）+ `human-sequences.bed`（chr/start/end/split 区间）。
- `HG38Dataset`（`src/dataloaders/datasets/hg38_dataset.py:126`）遍历 bed 区间，用 `pyfaidx` 查 FASTA，每样本**强制补齐/截断到 `max_length`**；next-token 目标由移位构造（`data=seq[:-1], target=seq[1:]`）。
- 分词：单核苷酸 `CharacterTokenizer`（`hg38_char_tokenizer.py`）——specials `[CLS][SEP][BOS][MASK][PAD][RESERVED][UNK]` + `A=7,C=8,G=9,T=10,N=11`，**vocab=12**，EOS=`[SEP]`，左 padding。
- 序列长度：1024 → 1M，通过 seqlen-warmup 逐步加长。
- 下游 loader（`genomics.py`）：GenomicBenchmarks、Nucleotide-Transformer(18)、染色质 profile(DeepSEA)、物种分类。

**训练逻辑（TRAINING）**
- 入口 `train.py`，框架 **PyTorch Lightning 1.8.6 + Hydra** 配置（`configs/` 分 experiment/model/dataset/pipeline）。
- 目标：因果 LM，`cross_entropy`；指标 PPL=`exp(mean NLL)`（`torchmetrics.py`）。
- 骨干 `ConvLMHeadModel`/`LMBackbone`（`long_conv_lm.py`）+ `HyenaOperator`（`hyena.py:270`，order-2 门控隐式 FFT 卷积），tied LM head。
- 超参：AdamW `lr=6e-4 wd=0.1`，cosine warmup，fp16，grad-clip 1.0。
- 启动：`python -m train experiment=hg38/hg38_hyena model.d_model=128 model.n_layer=2 dataset.max_length=1024 ...`

**评估逻辑（EVAL）**
- 下游用 `dna_embedding` + mean-pool `SequenceDecoder` 头，`load_backbone` 加载 backbone 重初始化头；或用 `huggingface.py` 加载 `LongSafari/*` 权重。
- 指标：MCC（NT enhancer/histone）、F1-macro（promoter/splice）、accuracy（GenomicBenchmarks）。
- ⚠️ **GUE 不在此 repo**（在 DNABERT-2）。

**复现坑**【已核验】：`src/` 训练路径**强制 import flash-attn**（需编译）；PL 1.8.6/transformers 4.26.1 版本锁死；`l_max=max_length+2` 且须与预训练模型一致；NT/GenomicBenchmarks 无 val split（val→test 别名）。

### 2.2 OpenToMe + flame + fla（复现落地平台）——【已核验，含本 repo 实测】

详见 `02_code_survey/02_opentome_flame_fla_analysis.md` + 本报告 §0。真实关系：

```
OpenToMe (本 checkout, 已扩展)
├── opentome/models/            ← HF 风格模型副本（可被 flame 训练）
│   ├── transformer/            ← Transformer++  ✅ 待复现目标
│   ├── delta_net/              ← DeltaNet        ✅ 待复现目标
│   ├── gated_deltanet/, gla/, gsa/, blt/, qwen3_next/
│   └── mergenet_nlp/           ← MergeNet ✅ HNet/MergeDNA 同族（Local+Latent+Local, DTEM 合并）
│       └── AutoModelForCausalLM.register(...)   ← 已 HF 注册
├── opentome/tokenizer/         ← blt/bytes/sentencepiece/tiktoken（byte 级，适配 DNA）
└── trainer/flame/              ← 内置 flame 训练引擎（torchtitan 底座）
    ├── flame/train.py          ← 按 BACKBONE 环境变量分发 import opentome.models.<x>
    ├── flame/data.py           ← HF datasets，在线读取 text/content 列并 tokenize
    ├── configs/*.json          ← transformer_340M / delta_net_340M / mergenet_64M/340M ...
    └── scripts/byte/*.sh        ← byte-level 训练启动模板（transformer++/deltanet/mergenet）
```

- **训练引擎**：`flame`（torchtitan 底座，支持 FSDP/TP/CP、`--training.varlen` 变长打包、`torch.compile`）；**模型**：`fla` + `opentome/models/` 提供 HF `PreTrainedModel`。
- **模型选择**：`export BACKBONE=delta_net_340M` + `--model.config configs/delta_net_340M.json`；`train.py` 据 BACKBONE 触发 `import opentome.models.delta_net`。
- **数据格式**【已核验 `flame/data.py`】：HF `datasets`，样本需含 `text` 或 `content` 字段，**在线 tokenize**，支持流式/多数据集 interleave/变长打包。→ **DNA 数据 = 一个含 `text` 列（核苷酸串）的 HF dataset。**
- **tokenizer**：默认走 HF `AutoTokenizer`（`--model.tokenizer_path`），或 `TOKENIZER_NAME=blt/bytes` 用内置 byte tokenizer（256 词表）。bos/eos 为变长打包所必需。
- **现状**：**全 repo 无任何 DNA 代码**（0 命中）。→ DNA 数据与评估需新写。

### 2.3 HNet / MergeNet 关系与可移植性——【已核验】

详见 `02_code_survey/03_hnet_benchmarks_analysis.md`。

- **goombalab 原版 HNet**（`repos/hnet`）：核心 `HNet`（`models/hnet.py`）与 `HNetForCausalLM`（`mixer_seq.py`）是干净的 `nn.Module`，动态分块在 `modules/dc.py`（`RoutingModule` 余弦相似度边界预测 + `ChunkLayer` 降采样 + `DeChunkLayer` EMA 升采样(复用 Mamba2 kernel) + 直通估计 STE）。**可移植性：模型高（干净 nn.Module），端到端复现中（无训练循环、无 DNA config、无 DNA checkpoint；依赖 mamba_ssm/causal_conv1d/flash_attn）。**
- **本 repo 的 MergeNet**（`opentome/models/mergenet_nlp`）：**同一分层思想的 HF 化实现**，已注册 AutoModel、已有 flame 配置与启动脚本、已跑通 byte-level 训练。→ **强烈建议以 MergeNet 为 HNet-spirit 的首选复现载体**，把原版 HNet 作为对照/严格复现的第二阶段。

---

## 3. 训练数据调研结论

| 用途 | 数据 | 来源 | 规模 | 备注 |
|---|---|---|---|---|
| **预训练**（主） | 人类基因组 **hg38** | `hg38.ml.fa`(Basenji GCS) + `human-sequences.bed` | ~3GB FASTA | HyenaDNA/Caduceus 同款；单核苷酸 |
| 预训练（可选扩展） | 多物种基因组 | NT/DNABERT-2 用 | 大 | 提升泛化，非必需 |
| **验证 PPL** | hg38 特定验证集 | bed 的 valid split | — | README 明确要求"特定验证集测 PPL" |
| 微调-基准① | **Genomic Benchmarks**（8 任务） | HyenaDNA 内置 dataloader / HF | 小 | Top-1 acc |
| 微调-基准② | **NT-bench**（18 任务） | HF `InstaDeepAI/nucleotide_transformer_downstream_tasks` | 中 | MCC/F1/acc，10-fold CV |
| 微调-基准③ | **GUE**（28 数据集/7 任务/4 物种） | DNABERT-2 repo（Google Drive zip） | 中 | MCC(主)/F1/acc，csv `sequence,label` |

**DNA→flame 数据格式转换**【推断，基于 §2.2 已核验的 data.py】：将 hg38 按 bed 区间切成定长核苷酸串，落成含 `text` 列的 HF dataset（arrow/parquet），即可喂给 flame。单核苷酸 tokenizer 有两种落地：(a) 造一个最小 HF `PreTrainedTokenizerFast`（A/C/G/T/N + bos/eos，vocab≈8）；(b) 直接用内置 `bytes` tokenizer（ACGTN 的 ASCII 天然是 byte，vocab=256）——**后者与 HNet/MergeNet 的 byte-level 训练范式天然契合**。

---

## 4. 评估 benchmark 调研结论

**双层指标体系**（与 README 一致）：

1. **验证性指标（过程）**：验证集 **PPL**（生成式 DNA 预训练的直接信号）。flame 训练中即可输出（`exp(mean NLL)`）。
2. **最终指标（结果）**：三大代表性 DNA 基准的下游微调分数。

| 基准 | 任务数 | 指标 | 数据获取 | 微调代码来源 |
|---|---|---|---|---|
| Genomic Benchmarks | 8 | Top-1 acc | HyenaDNA/Caduceus dataloader | HyenaDNA `src/` |
| NT-bench | 18（enhancer×2/histone×10/promoter×3/splice×3） | MCC / F1 / acc | HF datasets | HyenaDNA/Caduceus |
| GUE | 28（epigenetic×10/core-promoter×3/promoter×3/human-TF×5/mouse-TF×5/splice×1/covid×1） | MCC(主)/F1/acc | DNABERT-2 Google Drive | DNABERT-2 `finetune/train.py` |

**评估协议要点**【已核验】：生成式（因果 LM）模型评估下游时，取 backbone 输出做 mean-pool + 线性分类头微调（HyenaDNA 的 `dna_embedding` + `SequenceDecoder` 范式）；对 base-resolution 任务需保留解码器恢复分辨率（MergeDNA 的做法）。

---

## 5. 复现方案框架（基于调研的建议路线）

> 本节为**方案骨架**，非实现。README 已指定本次切换至 editable 模式、暂不实现。

**阶段化路线**（每阶段可独立验证）：

- **P0 环境**：建 `fla_environment.yml` 环境；`pip install -e OpenToMe`；编译 mamba-ssm/causal-conv1d/flash-attn（HNet/Mamba 依赖）。先跑通 `scripts/byte/mergenet.sh`（内置 SlimPajama 200 步 debug）验证训练链路。
- **P1 数据管线**：hg38 FASTA + bed → 含 `text` 列的 HF dataset（切窗、train/valid/test 按染色体分）；落地 DNA tokenizer（首选内置 `bytes`，vocab=256）。
- **P2 基线预训练**：用现成 **Transformer++** 与 **DeltaNet**（`configs/transformer_340M.json` / `delta_net_340M.json`）在 hg38 上跑因果 LM 预训练，监控验证 PPL。这是"已有模型复现"的最短路径（模型现成）。
- **P3 HNet 复现**：
  - 路径 A（推荐先做）：用现成 **MergeNet**（`mergenet_64M/340M.json`）作为 HNet-spirit 动态分块模型预训练，与 P2 基线对比 PPL。
  - 路径 B（严格复现）：将 goombalab 原版 HNet 按 flame `custom_models` 模式包 HF adapter（`PretrainedConfig`+`PreTrainedModel`），移植 `load_balancing_loss` 与分阶段 LR。
- **P4 评估接入**：接 GenomicBenchmarks → NT-bench → GUE 三级微调评估（复用 HyenaDNA 评估逻辑 + DNABERT-2 GUE 代码），产出最终指标表，横向对比 Transformer++ / DeltaNet / MergeNet(HNet)。

**主要风险 / 待确认项**：
- 【待确认】原版 HNet 的 CUDA kernel（mamba_ssm 版本）与 fla 栈的版本兼容性（`03_hnet_benchmarks_analysis.md` 标记为潜在冲突）。
- 【待确认】flame 的 pipeline parallelism 对 HNet/MergeNet 动态长度的支持（train.py:369 注释提示 PP+tie-embedding 未完全修好）——单机 FSDP 优先。
- 【推断】DNA byte tokenizer 下 varlen 打包需正确注入 bos/eos，否则序列边界错误。
- 【已核验】GUE 数据需科学上网从 Google Drive 下载（zip）。

---

## 6. 目录索引

```
dna_model_260703/
├── README.md                     ← 原始任务（用户提供）
├── 01_literature/
│   ├── INDEX.md                  ← 28 篇文献索引表（关键信息 + 本地路径）
│   └── papers/*.pdf              ← 28 篇已核验 PDF
├── 02_code_survey/
│   ├── 00_summary.md             ← 代码调研总览（本 repo 实测结论）
│   ├── 01_hyenadna_analysis.md   ← HyenaDNA 数据/训练/评估深度分析
│   ├── 02_opentome_flame_fla_analysis.md  ← OpenToMe/flame/fla 平台与迁移路径
│   ├── 03_hnet_benchmarks_analysis.md     ← HNet 可移植性 + GUE/NT-bench/Caduceus
│   └── repos/                    ← 8 个参考 repo（已删 .git）
│       ├── hyena-dna/ caduceus/ DNABERT_2/ nucleotide-transformer/
│       ├── OpenToMe/ flame/ flash-linear-attention/ hnet/
└── 03_report/
    └── survey_report.md          ← 本报告
```
