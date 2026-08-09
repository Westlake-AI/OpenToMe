# DNA 基础模型复现调研 · 入口文档

> **核心任务**：在 **OpenToMe**（`flash-linear-attention` + `flame` 训练引擎）中，参考开源 DNA 基础模型 repo，复现 **Transformer++ / DeltaNet** 与 **HNet** 在 human hg38 上的 **DNA 预训练 + 评估**。
> **当前阶段**：系统性调研已完成（文献 + 代码 + 数据 + 基准），**尚未进入实现**（editable 模式）。
> 整理日期：2026-07-03。本文件是本项目的总入口，串联「任务需求 → 调研结论 → 素材 → 下一步」。

---

## 0. 快速导航

| 你想… | 打开 |
|---|---|
| 读**完整调研报告**（结论 + 复现方案框架） | [`03_report/survey_report.md`](03_report/survey_report.md) |
| 查**文献**（28 篇 PDF + 关键信息表） | [`01_literature/INDEX.md`](01_literature/INDEX.md) |
| 查**参考代码仓库**（8 个 repo + 关键路径） | [`02_code_survey/INDEX.md`](02_code_survey/INDEX.md) |
| 看 **HyenaDNA / OpenToMe / HNet** 深度代码分析 | [`02_code_survey/00_summary.md`](02_code_survey/00_summary.md) + `01`–`03` |
| **重建**文献或代码（删了也能一键恢复） | 见下方 §5 |

---

## 1. 核心任务需求（原始要求，已整理）

### 1.1 模型开发目标
- **训练数据**：human **hg38**（参考 HyenaDNA 数据集）。
- **待复现模型**：
  - 已有模型：**Transformer++**、**DeltaNet**（fla 内置）。
  - 新模型：**HNet**（arXiv 2507.07955，Dynamic Chunking）。
- **复现载体**：**OpenToMe**（https://github.com/Westlake-AI/OpenToMe ，基于 `flash-linear-attention` + `flame`）。

### 1.2 评估目标（双层指标）
- **验证性指标**：特定验证集上的 **PPL**（生成式预训练的过程信号）。
- **最终指标**：三大代表性 DNA 基准的下游微调分数：
  1. **Nucleotide Transformer benchmark**（18 任务，参考 NT repo）
  2. **GUE benchmark**（参考 DNABERT-2 repo）
  3. **Genomic Benchmarks**（HyenaDNA 内置）

### 1.3 实验任务分解（原始）
1. 参考 repo：数据集配置 + 环境配置。
2. 参考 repo：预训练模型下载 → 模型评估 → 预训练复现。
3. 数据集 + 模型迁移：在 OpenToMe 中基于 flame 复现上述 repo 的评估与预训练。
4. 方法复现：在 OpenToMe 中复现 Transformer++ 和 HNet 在 hg38 上的预训练。

### 1.4 代码迁移思路（原始）
- 以 HyenaDNA repo 为核心，将 DNA 模型（DNABERT-2 / Hyena / Caduceus）的预训练代码迁移至 OpenToMe。
- 评估生成式 DNA 预训练模型：以验证集 PPL 为验证性指标，以代表性 benchmark 为最终指标。

---

## 2. 调研关键结论（TL;DR）

> 完整论证见 [`03_report/survey_report.md`](03_report/survey_report.md)。以下为最影响复现思路的 5 点。

1. **OpenToMe 现状比预期更有利。** 上游虽是 ViT token-merging 工具箱，但**公开仓库**已内置 LM 预训练超集：`trainer/flame/`（flame 训练引擎）+ `opentome/models/` 里 **Transformer++、DeltaNet、GatedDeltaNet、MergeNet** 等 HF 模型副本 + `scripts/byte/` byte-level 训练脚本。→ **Transformer++ 与 DeltaNet 已现成可训。**

2. **HNet 无需从零移植——repo 已有同族兄弟 MergeNet。** `opentome/models/mergenet_nlp` 是 Westlake 团队自研的分层 token-merging LM（Local + Latent + Local + DTEM 合并），已 `AutoModelForCausalLM.register`。它与 goombalab HNet、以及种子论文 **MergeDNA** 是同一思想的三个实现（MergeNet/MergeDNA 同出 Westlake-AI）。→ **优先用 MergeNet 复现 HNet 精神，严格版再包 HF adapter 移植原版 HNet。**

3. **HyenaDNA 是数据/评估权威蓝本，Caduceus 与之同源。** hg38 定长窗采样 + 单核苷酸 tokenizer（vocab=12）+ 因果 LM + PPL=exp(mean NLL) 全部可照搬。

4. **三大基准的数据与代码来源已定位。** GUE→DNABERT-2 repo（Google Drive + `finetune/train.py`）；NT-bench→HF `InstaDeepAI/nucleotide_transformer_downstream_tasks`；Genomic Benchmarks→HyenaDNA 内置。指标均为 MCC/F1/acc。

5. **数据侧与评估侧是真正要新写的部分。** OpenToMe 当前 0 行 DNA 代码；而 flame 数据契约就是「含 `text` 列的 HF dataset + 在线 tokenize」，DNA tokenizer 直接用内置 `bytes`（vocab=256，ACGTN 天然是 byte）即可，与 HNet/MergeNet byte-level 范式契合。

---

## 3. 素材地图

```
dna_model_260703/
├── README.md                       ← 本入口文档
├── 01_literature/                  ── 文献调研
│   ├── INDEX.md                    · 28 篇文献索引表（6 分组 + 关键信息 + 复现相关性）
│   ├── manifest.tsv                · 可重建数据源（下载真源）
│   ├── fetch_papers.sh             · 一键下载/校验脚本
│   └── papers/*.pdf                · 28 篇已核验 PDF
├── 02_code_survey/                 ── 代码调研
│   ├── INDEX.md                    · 8 个仓库索引（角色 + 关键路径）
│   ├── manifest.tsv                · 可重建数据源（克隆真源）
│   ├── clone_repos.sh              · 一键克隆脚本
│   ├── 00_summary.md               · 代码调研总览（本 repo 实测）
│   ├── 01_hyenadna_analysis.md     · HyenaDNA 数据/训练/评估深度分析
│   ├── 02_opentome_flame_fla_analysis.md · OpenToMe/flame/fla 平台 + 迁移路径
│   ├── 03_hnet_benchmarks_analysis.md    · HNet 可移植性 + GUE/NT-bench/Caduceus
│   └── repos/                      · 8 个参考仓库（已删 .git）
└── 03_report/
    └── survey_report.md            ← 主调研报告（结论 + 分阶段复现方案框架）
```

---

## 4. 复现方案框架（骨架，详见报告 §5）

> 本次仅调研、不实现。以下为分阶段路线，供后续决策。

- **P0 环境**：建 fla 环境 → `pip install -e OpenToMe` → 编译 mamba-ssm/causal-conv1d/flash-attn → 跑通 `scripts/byte/mergenet.sh` debug 验证链路。
- **P1 数据管线**：hg38 FASTA + bed → 含 `text` 列的 HF dataset（按染色体切 train/valid/test）→ DNA tokenizer（首选内置 `bytes`）。
- **P2 基线预训练**：用现成 Transformer++ / DeltaNet（`configs/*_340M.json`）在 hg38 跑因果 LM，监控验证 PPL。
- **P3 HNet 复现**：路径 A 先用现成 MergeNet；路径 B 严格移植 goombalab HNet（flame `custom_models` + HF adapter）。
- **P4 评估接入**：Genomic Benchmarks → NT-bench → GUE 三级微调，产出横向对比表。

**主要风险**：HNet CUDA kernel 与 fla 栈版本兼容性；flame 对动态长度模型的并行支持（优先单机 FSDP）；varlen 打包需正确注入 bos/eos；GUE 数据需科学上网下载。

---

## 5. 如何重建素材（删了也能恢复）

两个目录各自**自包含可重建**：`manifest.tsv` 是唯一数据源，配套脚本读取它下载/克隆。

```bash
# —— 重建文献 PDF ——
cd 01_literature
bash fetch_papers.sh            # 只下载缺失的（已实测：删任一 PDF 可精确恢复）
bash fetch_papers.sh --check    # 只校验现有 PDF（file 类型 + pypdf 页数），不下载
bash fetch_papers.sh --force    # 全部重下

# —— 重建参考代码仓库 ——
cd 02_code_survey
bash clone_repos.sh             # 只克隆缺失的（--depth 1，删 .git）
bash clone_repos.sh --check     # 只查存缺
bash clone_repos.sh --force     # 删后重克隆
```

- 增删论文/仓库：直接编辑对应 `manifest.tsv` 一行即可，脚本自动生效。
- **例外**：4 篇 bioRxiv 论文（Evo2 / GROVER / GENA-LM / GPN-MSA）因 Cloudflare 人机校验无法脚本下载，`fetch_papers.sh` 会打印浏览器下载链接，手动放入 `papers/` 即可（见 manifest 中 `manual` 条目）。

---

## 6. 参考链接（原始）

- OpenToMe: https://github.com/Westlake-AI/OpenToMe
- HyenaDNA: https://github.com/HazyResearch/hyena-dna
- DNABERT-2: https://github.com/MAGICS-LAB/DNABERT_2
- Nucleotide Transformer: https://github.com/instadeepai/nucleotide-transformer
- Caduceus: https://github.com/kuleshov-group/caduceus
- HNet: https://github.com/goombalab/hnet ｜ 论文 https://arxiv.org/pdf/2507.07955
- MergeDNA（文献种子）: https://arxiv.org/abs/2511.14806
