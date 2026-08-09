# DNA预训练代码开发

我期望在OpenToMe下参考一个DNA基础模型的开源github repo来复现DNA预训练。现阶段需要先做一套调研，包括两方面信息整理：
(1) 文献调研。系统性调研代表性DNA开源模型和相关闭源模型的论文（arXiv等学术网站），可参考MergeDNA（https://arxiv.org/abs/2511.14806）来进行文献调研，将相关文献（pdf）放置到指定目录，并给出包含各种关键信息的索引表格以便查看。
(2) 代码调研。为了系统性复现HNet的DNA预训练实验，考虑以HyenaDNA repo为核心，复现HNet相关代码。请调研下HyenaDNA repo中采用的训练数据、训练逻辑、评估逻辑。
请你系统性的帮我制定
(3) 调研目标。结合我在_claude_discussion/README.md中罗列的大致内容和参考思路，进行系统性的调研。注意，我的核心任务是，在OpenToMe中复现起已有模型（e.g., Transformer++ / DeltaNet）和HNet的DNA预训练和评估。请围绕该目标进行调研后，撰写调研报告。

## 模型开发
* 训练数据：参考HyenaDNA的human hg38数据集
* 标准评估：
(1) Nucleotide Transformer benchmark（参考Nucleotide Transformer repo）
(2) GUE benchmark（参考DNABERT-2 repo）
(3) 特定验证集上测PPL

## DNA开源模型

* Nucleotide Transformer: https://github.com/instadeepai/nucleotide-transformer
* DNABERT-2: https://github.com/MAGICS-LAB/DNABERT_2
* HyenaDNA: https://github.com/HazyResearch/hyena-dna
* Caduceus: https://github.com/kuleshov-group/caduceus

## 实验任务

### 代码复现
1. 参考repo：数据集配置 + 环境配置
2. 参考repo：预训练模型下载 -> 模型评估 -> 模型预训练复现
3. 数据集 + 模型迁移：在OpenToMe中基于flame复现上述repo的模型评估和预训练
. 方法复现：在OpenToMe中复现Transformer++和HNet在DNA预训练数据集（human hg38）上的预训练

### 待复现模型
HNet：https://arxiv.org/pdf/2507.07955

### 代码迁移
* 考虑如何以HyenaDNA repo为核心，将DNA模型（DNABERT-2 / Hyena / Caduceus）的预训练代码迁移至OpenToMe（https://github.com/Westlake-AI/OpenToMe），该repo以flash-linear-attn + flame库为基础
* 考虑如何评估generative DNA预训练模型？以评估集上PPL为验证性指标，以DNA代表性benchmark为最终指标

