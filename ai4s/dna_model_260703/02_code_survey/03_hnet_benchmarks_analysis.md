# H-Net, GUE, and NT / Caduceus Code Survey

Survey to support reproducing **H-Net DNA pretraining** and standard **DNA evaluations
(Nucleotide Transformer benchmark, GUE)** inside the OpenToMe/flame framework.

Repos analyzed (all under `02_code_survey/repos/`):
- `hnet` — goombalab/hnet, the H-Net "Dynamic Chunking" model
- `DNABERT_2` — MAGICS-LAB/DNABERT_2, source of the GUE benchmark
- `caduceus` — kuleshov-group/caduceus, Mamba-based bidirectional DNA model
- `nucleotide-transformer` — instadeepai, source of the NT benchmark tasks
- (context) `hyena-dna`, `flame`, `OpenToMe`, `flash-linear-attention`

---

## PART A — H-Net architecture & DNA experiments (`hnet`)

### A.1 Core innovation: dynamic chunking / learned tokenization
H-Net ("Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", Hwang, Wang, Gu,
arXiv 2507.07955) replaces a **fixed external tokenizer** with a **learned, content-adaptive
chunking mechanism** trained end-to-end with the model. It operates directly on **raw bytes**
(`vocab_size: 256`, `ByteTokenizer`) and learns *where* to place chunk boundaries, so the
"tokenization" is differentiable and data-driven rather than a fixed BPE/k-mer scheme.

### A.2 Architecture — hierarchical encoder / main / decoder + chunking
Recursive hierarchy defined in `hnet/hnet/models/hnet.py` (`class HNet`). Each stage is either:
- **Non-innermost** (`arch_layout` length 3): `[encoder, main_network, decoder]`, where
  `main_network` is itself another `HNet` (recursion into the next stage), or
- **Innermost** (`arch_layout` length 1): a single `main_network`.

Forward pass of a non-innermost stage (`hnet.py:204-300`):
1. `encoder` (Isotropic stack) processes the fine-grained sequence.
2. `residual_proj` (fp32 Linear, zero-init) saves a residual.
3. **`RoutingModule`** (boundary predictor) computes a boundary probability per position from
   the cosine similarity between adjacent hidden states: `boundary_prob = (1 - cos_sim)/2`,
   argmax → `boundary_mask` (first position forced to a boundary).
4. **`ChunkLayer`** downsamples: keeps only boundary positions → shorter sequence.
5. `main_network` (next hierarchy stage) runs on the compressed sequence.
6. **`DeChunkLayer`** upsamples back to full length using an **EMA smoother implemented via the
   Mamba2 kernel** (`mamba_chunk_scan_combined`), weighted by boundary probabilities.
7. Residual recombination via a straight-through estimator: `out * STE(p) + residual`
   (`STE`/`ste_func` in `hnet.py:20-31`) so boundary decisions get gradient.
8. `decoder` (Isotropic stack) processes the restored sequence.

**Implementing files:**
- `hnet/hnet/models/hnet.py` — hierarchical `HNet` module, `HNetState`, STE.
- `hnet/hnet/modules/dc.py` — **dynamic chunking**: `RoutingModule` (boundary predictor),
  `ChunkLayer`, `DeChunkLayer`, plus their `.step()` inference variants and state dataclasses.
- `hnet/hnet/modules/isotropic.py` — `Isotropic`, the non-hierarchical Mamba2/Attention stack
  (a stage's encoder/decoder/innermost block). Parses layout strings like `T24`, `m4`.
- `hnet/hnet/modules/block.py` — `create_block` / `Block`: wraps `Mamba2` (`m`/`M`) or
  `CausalMHA` (`t`/`T`) mixer + optional SwiGLU MLP + RMSNorm.
- `hnet/hnet/modules/mha.py` — FlashAttention-based causal MHA (self + cross + kv-cache).
- `hnet/hnet/models/mixer_seq.py` — `HNetForCausalLM`: `nn.Embedding` → `HNet` backbone →
  `lm_head`. This is the LM wrapper; the H-Net itself is a pure `(B,L,D)->(B,L,D)` map.
- `hnet/hnet/models/config_hnet.py` — `HNetConfig`, `AttnConfig`, `SSMConfig`.

### A.3 Config system (JSON)
Configs are plain JSON deserialized into dataclasses (see `generate.py:17-34`). Key fields
(`config_hnet.py`): `arch_layout` (nested list encoding the hierarchy, e.g.
`["m4", ["T1m4", ["T27"], "m4T1"], "m4"]` for a 2-stage XL), `d_model` (per stage),
`d_intermediate` (FFN dim per stage; 0 = no FFN), `vocab_size` (default **256 = bytes**),
`ssm_cfg` (Mamba2: `d_conv`, `expand`, `d_state`, `chunk_size`), `attn_cfg`
(`num_heads`, `rotary_emb_dim`, `window_size`), `tie_embeddings`.
Layout letters: `m`=Mamba2 no-MLP, `M`=Mamba2+MLP, `t`=Attention no-MLP, `T`=Attention+MLP.

Provided configs (`hnet/configs/`): `hnet_1stage_L`, `hnet_1stage_XL`, `hnet_2stage_L`,
`hnet_2stage_XL`, `hnet_2stage_XL_chinese`, `hnet_2stage_XL_code`. **All use `vocab_size: 256`
(byte level).** There is **NO DNA-specific config file** in the repo, and grep for
`dna|genome|nucleotide` across the repo returns nothing — the only genome-relevant property is
that everything is byte/nucleotide-friendly by construction (256-vocab byte modeling). The
released checkpoints are text/code/Chinese (FineWeb-Edu, Pile-GitHub), **not DNA**.

### A.4 What the repo provides
- **Model definition + inference/generation only.** `generate.py` loads a `.pt` checkpoint +
  JSON config and does byte-level autoregressive sampling.
- **No training loop / no training script.** `hnet/hnet/utils/train.py` only provides helper
  functions (`load_balancing_loss` for the router's downsampling-ratio target, and
  `group_params` for per-stage LR multipliers) and is explicitly annotated *"This file is not
  used inside the HNet package"* — you must supply your own training loop.
- **Pretrained checkpoints:** on HuggingFace `cartesia-ai` (`hnet_1stage_L/XL`,
  `hnet_2stage_L/XL`, `_chinese`, `_code`), trained on FineWeb-Edu (100B) / Chinese / Pile-GitHub.

### A.5 DNA experiments in the paper
The arXiv paper runs DNA experiments, but **this repo ships neither the DNA data pipeline nor
DNA configs**. From the architecture, the DNA setup is **byte-/single-nucleotide level** (the
same 256-vocab `ByteTokenizer` path): H-Net's whole premise is to consume raw nucleotides and
*learn* chunking, so no k-mer/BPE tokenizer is used. Reproducing DNA pretraining therefore
requires bringing your own genome dataloader (e.g. hg38, like HyenaDNA/Caduceus) and training
loop; only the model + router loss are provided here.

### A.6 Dependencies (`hnet/pyproject.toml`)
- `torch>=2.5.1`, `triton>=3.2.0`
- **`mamba_ssm`** (pinned git commit) — required; `Mamba2` mixer *and* the `DeChunkLayer` EMA
  reuse `mamba_chunk_scan_combined` (`dc.py:9`).
- **`flash_attn==2.8.0.post2`** — required; used for MHA (`mha.py`) and `RMSNorm`/layer-norm
  Triton kernels imported throughout (`block.py`, `isotropic.py`), plus `GenerationMixin`.
- **`causal_conv1d`** (pinned git commit) — Mamba2 depsendency.
- `einops`, `optree`, `regex`, `omegaconf`.
Effectively CUDA + custom Triton kernels are mandatory; there is no pure-PyTorch fallback.

### A.7 Portability into flame — assessment
**Verdict: model code is cleanly portable; the surrounding infra is not, and there is no DNA
recipe to copy.**
- **Clean side:** `HNet` (`hnet.py`) is a well-encapsulated `nn.Module` with a standard
  `(B,L,D)->(B,L,D)` contract, and `HNetForCausalLM` (`mixer_seq.py`) cleanly separates
  embedding/backbone/lm_head. It is **not** entangled with any Lightning/Trainer object — no
  training loop coupling. It returns `(logits, bpred_output, ...)` where `bpred_output` carries
  the router outputs needed for the auxiliary loss. This maps naturally onto flame's
  torchtitan-style custom-model pattern (`flame/custom_models/`).
- **Friction points for flame:**
  1. **Kernel dependency:** hard requirement on `mamba_ssm` + `causal_conv1d` + `flash_attn`
     Triton kernels (CUDA-only). flame already builds on `fla`/`flash-linear-attention`, so the
     stack is compatible, but these exact packages must be installed.
  2. **Auxiliary loss + optimizer plumbing:** you must wire `load_balancing_loss` into the
     training step (per router output) and replicate the **per-stage LR multipliers**
     (`apply_lr_multiplier`/`group_params`, which stash `param._optim`) — flame's optimizer
     grouping needs adapting to honor these.
  3. **Variable-length/packed path:** H-Net supports either a `mask` (padded) or `cu_seqlens`
     packed mode; the packed path assumes flash-attn varlen. flame's data collation must feed
     one of these two interfaces.
  4. **No DNA dataloader/config** ships here — must be authored (byte tokenizer exists at
     `hnet/hnet/utils/tokenizers.py::ByteTokenizer`).
- **Bottom line:** lift `hnet/models/*` + `hnet/modules/*` as a custom model in flame, install
  the three kernel deps, port `load_balancing_loss` + per-stage LR handling into the flame
  training step, and supply a byte-level genome dataloader. Model portability is HIGH; end-to-end
  DNA-pretraining reproduction is MEDIUM effort because the training recipe/data are absent.

---

## PART B — GUE benchmark (`DNABERT_2`)

### B.1 What GUE is
GUE (Genome Understanding Evaluation) is a benchmark of **28 datasets across 7 task types and 4
species** (`README.md:6,43-45`). Enumerated from `finetune/scripts/run_dnabert2.sh`:

| # | Task type | Datasets (folder) | Count | model_max_length | epochs |
|---|-----------|-------------------|-------|------------------|--------|
| 1 | Epigenetic Marks Prediction (histone, yeast) | `GUE/EMP/{H3,H3K14ac,H3K36me3,H3K4me1,H3K4me2,H3K4me3,H3K79me3,H3K9ac,H4,H4ac}` | 10 | 128 | 3 |
| 2 | Core promoter detection (human) | `GUE/prom/{prom_core_all,prom_core_notata,prom_core_tata}` | 3 | 20 | 4 / 4 / 10 |
| 3 | Promoter detection (human) | `GUE/prom/{prom_300_all,prom_300_notata,prom_300_tata}` | 3 | 70 | 4 / 4 / 10 |
| 4 | Transcription factor prediction (human) | `GUE/tf/{0,1,2,3,4}` | 5 | 30 | 3 |
| 5 | Transcription factor prediction (mouse) | `GUE/mouse/{0,1,2,3,4}` | 5 | 30 | 5 (max 1000 steps) |
| 6 | Splice site prediction | `GUE/splice/reconstructed` | 1 | 80 | 5 |
| 7 | COVID variant classification (virus) | `GUE/virus/covid` | 1 | 256 | 8 |

Total = 10+3+3+5+5+1+1 = **28**. Species: human, mouse, yeast, virus.

### B.2 Data download
`README.md:45,132`: GUE is a single zip on Google Drive:
`https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-/view?usp=sharing`.
Set `export DATA_PATH=/path/to/GUE` so folders resolve as `$DATA_PATH/GUE/<task>/<dataset>`.
(DNABERT-2 pretraining data is a separate Drive link, `README.md:123`.)

### B.3 Evaluation / fine-tuning code
`DNABERT_2/finetune/train.py` (HuggingFace `Trainer`-based):
- **Model:** `AutoModelForSequenceClassification.from_pretrained(..., num_labels=..., trust_remote_code=True)`
  — i.e. a **classification head is added on top of the pretrained encoder** (`train.py:261`).
  Optional **LoRA** (`--use_lora`, `peft`) used for the large NT models.
- **Metrics** (`calculate_metric_with_sklearn`, `train.py:189-207`): **accuracy, macro-F1,
  Matthews correlation (MCC), precision, recall**. MCC is DNABERT-2's headline GUE metric.
- **Data format:** CSV with header. Single-sequence: `sequence,label`
  (`ACGT...,1`); sequence-pair also supported (`text1,text2,label`) — see
  `train.py:122-135` and `DNABERT_2/sample_data/{train,dev,test}.csv`. Splits: train on
  `train.csv`, early-select on `dev.csv`, final eval on `test.csv` (`train.py:248-256`,
  `README.md:162`). `--kmer -1` = BPE (DNABERT-2); `3..6` = k-mer (DNABERT-1).

### B.4 Launch commands
```bash
export DATA_PATH=/path/to/GUE
cd finetune
sh scripts/run_dnabert2.sh $DATA_PATH        # DNABERT-2 over all 28 GUE datasets
sh scripts/run_dnabert1.sh $DATA_PATH 3      # DNABERT-1, 3-mer (3..6)
sh scripts/run_nt.sh       $DATA_PATH 0      # Nucleotide Transformer (0..3 select model size)
```
Per-task invocation (from `run_dnabert2.sh`), e.g. histone H3:
```bash
python train.py --model_name_or_path zhihan1996/DNABERT-2-117M \
  --data_path $DATA_PATH/GUE/EMP/H3 --kmer -1 --model_max_length 128 \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 --num_train_epochs 3 --fp16 \
  --evaluation_strategy steps --eval_steps 200 --warmup_steps 50 \
  --output_dir output/dnabert2 --overwrite_output_dir True
```
Global batch size is tuned to 32 (paper setting); switch `python`→`torchrun --nproc_per_node N`
for DDP. NT runs (`run_nt.sh`) add `--use_lora --lora_target_modules 'query,value,key,dense'`,
lr `1e-4`.

---

## PART C — Caduceus & Nucleotide Transformer benchmark

### C.1 Caduceus
- **Backbone: bidirectional Mamba ("MambaDNA"/BiMamba).** `caduceus/caduceus/modeling_caduceus.py`
  builds blocks from `mamba_ssm`'s `Mamba` wrapped in `BiMambaWrapper` (a forward + reverse Mamba,
  optionally weight-tied on in/out projections; `bidirectional_strategy="add"`).
  `CaduceusConfig` (`configuration_caduceus.py`) extends MambaConfig with
  `bidirectional`, `bidirectional_strategy`, `bidirectional_weight_tie`, and `rcps`.
- **Reverse-complement (RC) equivariance:** `caduceus/caduceus/modeling_rcps.py` provides
  `RCPSEmbedding`, `RCPSLMHead`, `RCPSAddNormWrapper`, `RCPSMambaBlock` (RC-Parameter-Sharing).
  Two variants: **Caduceus-PS** (RC-equivariant by construction, `rcps=true`, no RC aug needed)
  and **Caduceus-Ph** (RC data-augmentation, "post-hoc" conjoining at test). RC handling is
  visible throughout the dataloaders as `rc_aug` / `conjoin_train` / `conjoin_test`.

- **Built on the HyenaDNA / Safari codebase — CONFIRMED.** `caduceus/README.md:268-271` states
  the repo *"is adapted from the HyenaDNA repo and leverages much of the training, data loading,
  and logging infrastructure... HyenaDNA was originally derived from S4 and Safari."* Concrete
  evidence in this checkout:
  - Same Hydra config tree (`configs/experiment/hg38/*`, `configs/pipeline/*`,
    `configs/model/{hyena,mamba,caduceus}.yaml`), same `train.py` entry point, same
    `src/{dataloaders,tasks,models}` layout as `hyena-dna/`.
  - `caduceus/src/dataloaders/datasets/nucleotide_transformer_dataset.py` is a near-identical
    fork of `hyena-dna/src/dataloaders/datasets/nucleotide_transformer_dataset.py` (diffed:
    Caduceus switches the source to the HF dataset + adds `conjoin_train/conjoin_test` for RC;
    logic otherwise the same).
  - Hyena operator itself is carried in `caduceus/src/models/sequence/hyena.py`.
  **Implication:** Caduceus's DNA data/eval pipeline (hg38 pretraining, GenomicBenchmarks, NT
  tasks, char tokenizer, cross-validation eval) **mirrors HyenaDNA exactly**, so a flame port can
  reuse the HyenaDNA-style data recipe.

- **Downstream evaluations:** (1) **Nucleotide Transformer** 18 tasks
  (`configs/experiment/hg38/nucleotide_transformer.yaml`), (2) **GenomicBenchmarks** 8 tasks
  (`genomic_benchmark`), (3) **eQTL SNP Variant Effect Prediction** from the Long-Range Benchmark
  (embed with `vep_embeddings.py`, then SVM in `vep_svm.ipynb`). Fine-tuning uses a mean-pool +
  linear decoder head (`dna_embedding`/`dna_embedding_caduceus`), with `conjoin_*` flags feeding
  RC channels to the decoder for the PS variant.

### C.2 Nucleotide Transformer benchmark — the 18 downstream tasks
Data lives on HuggingFace **`InstaDeepAI/nucleotide_transformer_downstream_tasks`**, loaded
directly via `datasets.load_dataset(...)` in
`caduceus/src/dataloaders/datasets/nucleotide_transformer_dataset.py:53-57`. The task list,
class counts, sequence lengths and **per-task metrics** are enumerated in
`caduceus/configs/dataset/nucleotide_transformer.yaml` (also the wrapper task loop in
`slurm_scripts/wrapper_run_nucleotide_transformer.sh`):

| Group | Task (`dataset_name`) | classes | max_len | metric |
|-------|-----------------------|---------|---------|--------|
| Enhancers | `enhancers` | 2 | 200 | **MCC** |
| Enhancers | `enhancers_types` | 3 | 200 | **MCC** |
| Histone marks | `H3` | 2 | 500 | **MCC** |
| Histone marks | `H3K4me1` | 2 | 500 | **MCC** |
| Histone marks | `H3K4me2` | 2 | 500 | **MCC** |
| Histone marks | `H3K4me3` | 2 | 500 | **MCC** |
| Histone marks | `H3K9ac` | 2 | 500 | **MCC** |
| Histone marks | `H3K14ac` | 2 | 500 | **MCC** |
| Histone marks | `H3K36me3` | 2 | 500 | **MCC** |
| Histone marks | `H3K79me3` | 2 | 500 | **MCC** |
| Histone marks | `H4` | 2 | 500 | **MCC** |
| Histone marks | `H4ac` | 2 | 500 | **MCC** |
| Promoters | `promoter_all` | 2 | 300 | **F1** (binary) |
| Promoters | `promoter_no_tata` | 2 | 300 | **F1** (binary) |
| Promoters | `promoter_tata` | 2 | 300 | **F1** (binary) |
| Splice | `splice_sites_all` | 3 | 400 | **accuracy** |
| Splice | `splice_sites_acceptors` | 2 | 600 | **F1** (binary) |
| Splice | `splice_sites_donors` | 2 | 600 | **F1** (binary) |

= **18 tasks** (2 enhancer + 10 histone + 3 promoter + 3 splice). Metric functions live in
`caduceus/src/tasks/metrics.py` (`mcc`, `f1_binary`, `accuracy`). Evaluation protocol
(`nucleotide_transformer.yaml`): 10-fold **cross-validation** (`train.cross_validation=true`),
up to 20 epochs, char tokenizer, mean-pool linear decoder.

**Note on the `nucleotide-transformer` repo itself:** it is primarily an *inference/model-zoo*
repo (JAX/Haiku + PyTorch model defs for NT, NTv3, SegmentNT, ChatNT, Enformer, Borzoi, etc. under
`nucleotide_transformer/`) — it does **not** contain the 18-task fine-tuning harness. The
canonical way to pull the benchmark is the HF dataset above (as DNABERT-2, HyenaDNA, and Caduceus
all do); the leaderboard is `InstaDeepAI/nucleotide_transformer_benchmark`.

---

## Cross-cutting notes for the flame reproduction

- **Two eval suites, two data sources, two tokenizations:**
  - **GUE** → Google-Drive zip, CSV `sequence,label`, HF-`Trainer` fine-tune with a classification
    head; primary metric **MCC** (also F1/acc). Reference impl: `DNABERT_2/finetune/train.py`.
  - **NT benchmark** → HF dataset `InstaDeepAI/nucleotide_transformer_downstream_tasks`, char/byte
    tokenization, mean-pool+linear decoder, 10-fold CV; metrics **MCC / F1 / accuracy** per the
    table above. Reference impl: Caduceus (= HyenaDNA) `train.py` + `src/`.
- **H-Net fits the NT/HyenaDNA style naturally** (byte-level, single-nucleotide, learned chunking),
  so the Caduceus/HyenaDNA NT dataloader + decoder head can be reused around an H-Net backbone
  ported into flame; GUE can be reproduced by feeding the same CSV data through H-Net + a linear
  classification head and reporting MCC.
- **Shared kernel stack** (`mamba_ssm`, `causal_conv1d`, `flash_attn`, Triton) is required by both
  H-Net and Caduceus, and is compatible with flame's `fla`/`flash-linear-attention` base.

### Key file paths
- H-Net model: `repos/hnet/hnet/models/hnet.py`, `.../mixer_seq.py`, `.../config_hnet.py`
- H-Net chunking: `repos/hnet/hnet/modules/dc.py`, `.../isotropic.py`, `.../block.py`, `.../mha.py`
- H-Net loss/optim helpers: `repos/hnet/hnet/utils/train.py`; byte tokenizer: `.../utils/tokenizers.py`
- H-Net configs/deps: `repos/hnet/configs/*.json`, `repos/hnet/pyproject.toml`
- GUE fine-tune: `repos/DNABERT_2/finetune/train.py`, `.../scripts/run_dnabert2.sh`, `.../run_nt.sh`; data fmt `.../sample_data/*.csv`; download in `repos/DNABERT_2/README.md`
- Caduceus model: `repos/caduceus/caduceus/modeling_caduceus.py`, `.../modeling_rcps.py`, `.../configuration_caduceus.py`
- Caduceus↔HyenaDNA NT loader: `repos/caduceus/src/dataloaders/datasets/nucleotide_transformer_dataset.py` (cf. `repos/hyena-dna/src/dataloaders/datasets/nucleotide_transformer_dataset.py`)
- NT task list/metrics: `repos/caduceus/configs/dataset/nucleotide_transformer.yaml`, `.../configs/experiment/hg38/nucleotide_transformer.yaml`, `.../src/tasks/metrics.py`, `.../slurm_scripts/wrapper_run_nucleotide_transformer.sh`
