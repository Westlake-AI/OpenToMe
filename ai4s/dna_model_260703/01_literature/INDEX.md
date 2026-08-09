# 文献索引 — DNA 基础模型复现调研

> 下载日期：2026-07-03。本地根目录：`01_literature/papers/`
> 所有 arXiv ID 均已通过 arXiv API 核对标题；期刊/bioRxiv 论文通过 Europe PMC / 出版社开放获取（OA）渠道下载并用 pypdf 校验页数。
> 选文范围以 **MergeDNA (arXiv 2511.14806)** 的相关工作 + 参考文献为种子扩充，覆盖四类模型范式（SSM / Transformer / Hybrid / CNN）+ 动态分词方法 + byte-level 骨干 + 基准。
> **★ = 与本次核心任务（OpenToMe 复现 Transformer++/DeltaNet/HNet 的 DNA 预训练）最相关。**

> ### 🔁 可重建（即使删除 PDF 也能一键恢复）
>
> - **数据源**：`manifest.tsv`（filename / type / source / group / title），是下载的唯一真源；编辑此表即可增删论文。
> - **脚本**：`bash fetch_papers.sh`（只下缺失）｜`--force`（全部重下）｜`--check`（只校验不下载）。
> - **验证**：脚本内置 `file` 类型检查 + pypdf 页数校验；已实测「删除任一 PDF → 重跑 → 精确恢复」。
> - **manual 条目**：4 篇 bioRxiv 论文（Evo2 / GROVER / GENA-LM / GPN-MSA）因 Cloudflare 拦截无法脚本下载，脚本会打印浏览器下载链接，手动放入 `papers/` 即可。

---

## A. 代表性开源 DNA 模型


| 资料名                          | 类型     | venue+年             | ID/DOI                             | 本地路径                                                          | 骨干                                    | 分词              | 参数量       | 预训练数据                        | 评估基准                                | 复现相关性                         |
| ---------------------------- | ------ | ------------------- | ---------------------------------- | ------------------------------------------------------------- | ------------------------------------- | --------------- | --------- | ---------------------------- | ----------------------------------- | ----------------------------- |
| ★ **HyenaDNA**               | 开源-DNA | NeurIPS 2023        | arXiv 2306.15794                   | papers/hyenadna_2306.15794.pdf                                | Hyena（隐式长卷积，decoder/因果 LM）            | 单核苷酸 char       | 0.4M–6.6M | 人类 hg38（全基因组）                | GenomicBenchmarks, NT-bench, 染色质    | **核心参考 repo**；数据/评估管线的蓝本      |
| ★ **Caduceus**               | 开源-DNA | ICML 2024           | arXiv 2403.03234                   | papers/caduceus_2403.03234.pdf                                | BiMamba（RC 等变，MambaDNA）               | 单核苷酸 char       | 1.9M–7.7M | 人类 hg38                      | NT-bench, GenomicBenchmarks, 长程 VEP | 代码 fork 自 HyenaDNA，配置/数据/评估同源 |
| ★ **DNABERT-2**              | 开源-DNA | ICLR 2024           | arXiv 2306.15006                   | papers/dnabert2_2306.15006.pdf                                | Transformer(+ALiBi+FlashAttn) encoder | BPE             | ~117M     | 多物种（135 种）                   | **GUE 基准提出方**（28 数据集/7 任务）          | GUE 评估的数据+微调代码来源              |
| DNABERT (v1)                 | 开源-DNA | Bioinformatics 2021 | DOI 10.1093/bioinformatics/btab083 | papers/dnabert_bioinformatics_2021_PMC11025658.pdf            | BERT-base encoder                     | k-mer (3/4/5/6) | ~110M     | 人类 hg38                      | 启动子/剪接/TFBS                         | k-mer 分词与早期基线                 |
| ★ **Nucleotide Transformer** | 开源-DNA | Nature Methods 2024 | DOI 10.1038/s41592-024-02523-z     | papers/nucleotide_transformer_natmethods_2024_PMC11810778.pdf | BERT/ESM-style encoder                | 6-mer（非重叠）      | 50M–2.5B  | 人类+1000G+多物种(850)            | **NT-bench 提出方**（18 任务）             | NT-bench 下游任务与指标来源            |
| GPN                          | 开源-DNA | PNAS 2023           | DOI 10.1073/pnas.2311219120        | papers/gpn_pnas_2023_PMC10622914.pdf                          | 膨胀卷积网络                                | 单核苷酸            | ~30–66M   | 拟南芥/十字花科基因组                  | 变异效应预测（植物）                          | 单核苷酸卷积基线、变异效应零样本              |
| Evo                          | 开源-DNA | Science 2024        | DOI 10.1126/science.ado9336        | papers/evo_science_2024_PMC12057570.pdf                       | StripedHyena（Hyena+attn 混合）           | 单核苷酸 byte       | 7B        | OpenGenome（原核/噬菌体 ~300B tok） | 零样本 fitness、生成、基因必需性                | byte-level 生成式 DNA LM 代表      |
| DNAGPT                       | 开源-DNA | arXiv 2023          | arXiv 2307.05628                   | papers/dnagpt_2307.05628.pdf                                  | GPT decoder + 数值头                     | k-mer           | 0.1B–3B   | 多物种基因组                       | 基因组信号、mRNA/表达                       | 生成式 GPT 式 DNA LM              |
| HybriDNA                     | 开源-DNA | arXiv 2025          | arXiv 2502.10807                   | papers/hybridna_2502.10807.pdf                                | Transformer-Mamba2 混合                 | 单核苷酸            | 300M–7B   | 多物种长序列                       | GUE, NT-bench, 长程生成                 | Hybrid 范式（attn+SSM）长程 DNA     |
| ConvNova                     | 开源-DNA | ICLR 2025           | arXiv 2502.18538                   | papers/convnova_2502.18538.pdf                                | 现代 CNN 架构                             | 单核苷酸            | ~1.7M     | 人类 hg38                      | GenomicBenchmarks, NT-bench         | CNN 范式重新审视（轻量强基线）             |


## B. DNA 动态分词 / 自适应词表方法（★ HNet 的 DNA 直接对标）


| 资料名            | 类型     | venue+年      | ID               | 本地路径                           | 机制                                                                  | 复现相关性                                                              |
| -------------- | ------ | ------------ | ---------------- | ------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------ |
| ★ **MergeDNA** | 开源-DNA | arXiv 2025   | arXiv 2511.14806 | papers/mergedna_2511.14806.pdf | 可微 Token Merging + 分层（Latent Encoder/Decoder + Local Decoder），自适应分词 | **本调研的种子论文**；直接对标 HNet 的"学习式分词"思想在 DNA 上的落地；D=1024，local window=16 |
| VQDNA          | 开源-DNA | ICML 2024    | arXiv 2405.10812 | papers/vqdna_2405.10812.pdf    | 向量量化（VQ）学习基因组词表                                                     | 学习式离散分词，替代 k-mer/BPE                                               |
| MxDNA          | 开源-DNA | NeurIPS 2024 | arXiv 2412.13716 | papers/mxdna_2412.13716.pdf    | 模型自决定分词（可微稀疏卷积路由）                                                   | 与 HNet 动态分块高度同构（DNA 域）                                             |
| Omni-DNA       | 开源-DNA | arXiv 2025   | arXiv 2502.03499 | papers/omnidna_2502.03499.pdf  | 统一跨模态/多任务基因组基础模型                                                    | 多任务评估范式参考                                                          |


## C. 序列建模骨干与方法（★ OpenToMe 内的复现目标）


| 资料名                           | 类型    | venue+年      | ID               | 本地路径                                  | 机制                                                                             | 复现相关性                                         |
| ----------------------------- | ----- | ------------ | ---------------- | ------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------- |
| ★ **HNet (Dynamic Chunking)** | 骨干-方法 | arXiv 2025   | arXiv 2507.07955 | papers/hnet_2507.07955.pdf            | 分层 encoder-main-decoder + 学习式动态分块（RoutingModule/ChunkLayer/DeChunkLayer + STE） | **待复现模型**；论文含 byte-level DNA (HG38) 实验        |
| ★ **DeltaNet**                | 骨干-方法 | NeurIPS 2024 | arXiv 2406.06484 | papers/deltanet_2406.06484.pdf        | 线性注意力 + delta rule（分块并行）                                                       | **待复现模型**，fla 已内置 `delta_net`                 |
| ★ **Gated DeltaNet**          | 骨干-方法 | ICLR 2025    | arXiv 2412.06464 | papers/gated_deltanet_2412.06464.pdf  | 门控 delta rule（Mamba2+delta）                                                    | fla 已内置 `gated_deltanet`，DeltaNet 增强版         |
| ★ **Mamba**                   | 骨干-方法 | COLM 2024    | arXiv 2312.00752 | papers/mamba_2312.00752.pdf           | 选择性 SSM                                                                        | Caduceus/HybriDNA 骨干；fla 内置；HNet 依赖 mamba-ssm |
| ★ **Mamba-2**                 | 骨干-方法 | ICML 2024    | arXiv 2405.21060 | papers/mamba2_2405.21060.pdf          | SSD（状态空间对偶）                                                                    | HNet 的 DeChunkLayer 复用其 kernel；fla 内置         |
| Hyena Hierarchy               | 骨干-方法 | ICML 2023    | arXiv 2302.10866 | papers/hyena_hierarchy_2302.10866.pdf | 隐式长卷积 + 门控                                                                     | HyenaDNA/Evo 的骨干原始论文                          |


## D. Byte-level / 无分词分层建模（★ HNet 的 NLP 同族方法，思想借鉴）


| 资料名                           | 类型    | venue+年    | ID               | 本地路径                                              | 机制                             | 复现相关性                                        |
| ----------------------------- | ----- | ---------- | ---------------- | ------------------------------------------------- | ------------------------------ | -------------------------------------------- |
| Byte Latent Transformer (BLT) | 骨干-方法 | arXiv 2024 | arXiv 2412.09871 | papers/blt_byte_latent_transformer_2412.09871.pdf | 基于熵的动态 patch（byte→latent）      | HNet 之前最接近的"学习式分块"；OpenToMe 内置 BLT tokenizer |
| MambaByte                     | 骨干-方法 | COLM 2024  | arXiv 2401.13660 | papers/mambabyte_2401.13660.pdf                   | byte-level 选择性 SSM             | 无分词 SSM，DNA byte 建模直接可借                      |
| MEGABYTE                      | 骨干-方法 | arXiv 2023 | arXiv 2305.07185 | papers/megabyte_2305.07185.pdf                    | 多尺度 patch Transformer（百万 byte） | 固定 patch 分层，HNet 的前身思想                       |
| SpaceByte                     | 骨干-方法 | arXiv 2024 | arXiv 2404.14408 | papers/spacebyte_2404.14408.pdf                   | 基于边界（空格）的动态 patch              | 规则式动态分块对照                                    |
| Dynamic Token Pooling         | 骨干-方法 | arXiv 2022 | arXiv 2211.09761 | papers/dynamic_token_pooling_2211.09761.pdf       | 学习式 token 池化边界                 | HNet Routing 的早期思想来源                         |


## E. 闭源 / 标志性基因组模型（背景对照）


| 资料名      | 类型     | venue+年             | ID/DOI                         | 本地路径                                | 机制                                  | 复现相关性             |
| -------- | ------ | ------------------- | ------------------------------ | ----------------------------------- | ----------------------------------- | ----------------- |
| Enformer | 闭源-DNA | Nature Methods 2021 | DOI 10.1038/s41592-021-01252-x | papers/enformer_natmethods_2021.pdf | Conv trunk + Transformer（200kb 上下文） | 表达/表观预测标杆；长程上下文对照 |


## F. 评估基准论文


| 资料名                      | 类型  | venue+年                  | ID/DOI                         | 本地路径                                   | 内容               | 复现相关性                                                       |
| ------------------------ | --- | ------------------------ | ------------------------------ | -------------------------------------- | ---------------- | ----------------------------------------------------------- |
| ★ **Genomic Benchmarks** | 基准  | BMC Genomic Data 2023    | DOI 10.1186/s12863-023-01123-8 | papers/genomic_benchmarks_bmc_2023.pdf | 8 类基因组序列分类任务     | HyenaDNA/Caduceus 主用下游基准之一                                  |
| GenBench                 | 基准  | arXiv 2024               | arXiv 2406.01627               | papers/genbench_2406.01627.pdf         | 基因组基础模型系统评估套件    | 补充评估视角                                                      |
| （GUE 基准）                 | 基准  | 见 DNABERT-2              | arXiv 2306.15006               | 同 DNABERT-2                            | 28 数据集/7 任务/4 物种 | **GUE 数据+微调代码在 DNABERT-2 repo**                             |
| （NT-bench）               | 基准  | 见 Nucleotide Transformer | DOI 10.1038/s41592-024-02523-z | 同 NT                                   | 18 下游任务          | **HF: InstaDeepAI/nucleotide_transformer_downstream_tasks** |


---

## MergeDNA 相关工作梳理（种子论文的文献地图）

MergeDNA 将 DNA 基础模型按 **4 种架构范式 + 4 种分词类型** 归类，本索引据此扩充：

- **范式①（SSM）**：HyenaDNA, Caduceus ✅
- **范式②（Transformer）**：DNABERT, DNABERT-2, NTv1/v2, GROVER, GenSLM ✅（GenSLM/GROVER 见下方"未下载"）
- **范式③（Hybrid）**：Evo, Evo2, HybriDNA ✅
- **范式④（CNN）**：ConvNova ✅
- **分词类型**：byte-level（Evo）、k-mer（NTv2）、BPE（DNABERT-2）、**动态分词（VQDNA / MxDNA / MergeDNA）** ✅ ← 与 HNet 动态分块同族

MergeDNA 评估用三大基准：**Genomic Benchmarks（8 任务，Top-1 acc）**、**GUE（24/28 任务，MCC/F1）**、**NT-bench（18 任务）**，与本项目 README 目标基准完全一致。

## 未下载 / 无开放 PDF（仅记录，供后续手动获取）


| 资料名                       | 原因                                          | 获取建议                                                                  |
| ------------------------- | ------------------------------------------- | --------------------------------------------------------------------- |
| Evo 2 (Brixi et al. 2025) | Arc Institute manuscript，无稳定 arXiv/OA       | github.com/ArcInstitute/evo2；bioRxiv 2025.02.18.638918（Cloudflare 拦截） |
| GROVER (Sanabria 2023)    | bioRxiv 2023.07.19.549677，Cloudflare 拦截脚本下载 | 浏览器手动下载 bioRxiv v2                                                    |
| GenSLM (Zvyagin 2022)     | bioRxiv，脚本受限                                | 手动获取                                                                  |
| GENA-LM (Fishman/Ji 2023) | bioRxiv/NAR                                 | github.com/AIRI-Institute/GENA_LM                                     |
| GPN-MSA (Benegas 2023)    | bioRxiv 2023.10.10.561776                   | 手动获取                                                                  |


> 注：bioRxiv 全站启用 Cloudflare 人机校验，命令行 `curl` 无法直接取 PDF；期刊 OA 版本（Nature/PNAS/BMC/PMC）已通过 Europe PMC render 渠道成功获取。上述剩余 5 篇如需可用浏览器手动补齐到 `papers/`。

