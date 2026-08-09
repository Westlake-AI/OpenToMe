# 代码仓库索引 — DNA 模型复现参考

> 克隆日期：2026-07-03。本地根目录：`02_code_survey/repos/`（均已删 `.git`，浅克隆）。
> 深度分析见同目录 `00_summary.md` 与 `01`–`03` 分析文档。
> **★ = 与核心任务（OpenToMe 复现 Transformer++/DeltaNet/HNet 的 DNA 预训练+评估）最相关。**

## 🔁 可重建（即使删除仓库也能一键恢复）
- **数据源**：`manifest.tsv`（dirname / git_url / role / note），是克隆的唯一真源；编辑此表即可增删仓库。
- **脚本**：`bash clone_repos.sh`（只克隆缺失）｜`--force`（删后重克隆）｜`--keep-git`（保留 .git）｜`--check`（只查存缺）。
- 默认 `--depth 1` 浅克隆 + `GIT_LFS_SKIP_SMUDGE=1` 跳过大权重文件 + 删 `.git`（省空间）。已实测 8/8 仓库 `--check` 全 OK。

## 仓库清单

| 仓库 | 角色 | 上游 | 复现价值 & 关键路径 |
|---|---|---|---|
| ★ **hyena-dna** | 蓝本 | HazyResearch/hyena-dna | **DNA 数据/训练/评估蓝本**。`src/dataloaders/datasets/hg38_dataset.py`（HG38 定长窗采样）、`src/dataloaders/datasets/hg38_char_tokenizer.py`（单核苷酸 tokenizer, vocab=12）、`train.py`（PL+Hydra 因果 LM）、`src/tasks/metrics.py`（PPL/MCC/F1） |
| ★ **caduceus** | 蓝本 | kuleshov-group/caduceus | fork 自 hyena-dna，配置/数据/评估同源；`src/models/sequence/`（BiMamba）、`configs/dataset/nucleotide_transformer.yaml`（NT-bench 接入范例） |
| ★ **DNABERT_2** | 基准 | MAGICS-LAB/DNABERT_2 | **GUE 基准来源**。`README.md`（GUE Google Drive 下载）、`finetune/train.py`（分类头微调, MCC/F1/acc）、`scripts/run_dnabert2.sh` |
| **nucleotide-transformer** | 基准 | instadeepai/nucleotide-transformer | NT-bench(18 任务) 模型库；下游数据在 HF `InstaDeepAI/nucleotide_transformer_downstream_tasks` |
| ★ **OpenToMe** | 平台 | Westlake-AI/OpenToMe | **复现平台**。`trainer/flame/{configs/*.json,scripts/byte/*.sh,flame/train.py,flame/data.py}`、`opentome/models/{transformer,delta_net,gated_deltanet,mergenet_nlp}`、`opentome/tokenizer/{blt,bytes}`。注：`trainer/flame`+`mergenet_nlp` 在**公开仓库**即有（非本地私改） |
| **flame** | 平台 | fla-org/flame | 训练引擎(torchtitan 底座) 上游参考；OpenToMe 内已 vendored 一份 |
| ★ **flash-linear-attention** | 平台 | fla-org/flash-linear-attention | `fla/models/{transformer,delta_net,gated_deltanet,mamba,mamba2,...}`（HF `PreTrainedModel`，供 flame 训练） |
| ★ **hnet** | 待复现 | goombalab/hnet | **HNet 原版**。`hnet/models/hnet.py`（HNetForCausalLM）、`hnet/modules/dc.py`（RoutingModule/ChunkLayer/DeChunkLayer 动态分块）、`generate.py`（仅推理，无训练循环） |

## 复现关键结论（详见 00_summary.md）
1. **Transformer++ / DeltaNet 在 OpenToMe 已现成可训**（`opentome/models/` + `trainer/flame/configs/*_340M.json` + `scripts/byte/*.sh`）。
2. **HNet 有同族兄弟 MergeNet**（`opentome/models/mergenet_nlp`，已 `AutoModelForCausalLM.register`）—— 与 goombalab HNet、MergeDNA 同思想（均出 Westlake-AI）。优先用 MergeNet，严格复现再包 HNet HF adapter。
3. **数据/评估侧需新写**：OpenToMe 当前 0 行 DNA 代码；hg38→含 `text` 列的 HF dataset + DNA tokenizer（首选内置 `bytes`），评估复用 hyena-dna 逻辑 + DNABERT_2 的 GUE 代码。
