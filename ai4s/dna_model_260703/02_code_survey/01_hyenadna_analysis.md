# HyenaDNA Technical Report: Data Pipeline, Training Logic, Evaluation Logic

Repo analyzed: `/Users/lisiyuan/Downloads/local_exp/_claude_discussion/dna_model_260703/02_code_survey/repos/hyena-dna`

HyenaDNA is a long-range genomic foundation model pretrained on the human reference genome (hg38) at single-nucleotide resolution with context lengths up to ~1M tokens. It is built on the **Safari / S4** research codebase: **PyTorch Lightning** for the training loop + **Hydra/OmegaConf** for configuration, with the **Hyena** implicit long-convolution operator as the sequence mixer and a GPT-2-style causal LM head. The pretraining objective is **next-token (causal) language modeling** over DNA characters.

---

## 1. DATA

### 1.1 hg38 pretraining dataset

**Config:** `configs/dataset/hg38.yaml`
```yaml
_name_: hg38
bed_file: null          # defaults to data/hg38/human-sequences.bed
fasta_file: null        # defaults to data/hg38/hg38.ml.fa
max_length: 1024
add_eos: True
batch_size: 8           # per GPU
__train_len: ${div_up:1_000_000_000, ${.max_length}}   # ~1B tokens / seq_len = #samples per epoch
__l_max: ${.max_length}
```
Note `__train_len` = `ceil(1e9 / max_length)`: one "epoch" is defined as ~1 billion nucleotides worth of samples. This is what the scheduler uses to compute total steps.

**Dataloader (LightningDataModule):** `HG38` class in `src/dataloaders/genomics.py:29-215`.
- `_name_ = "hg38"` (line 44) is the registry key the dataset config resolves to.
- Default file paths set in `__init__` (lines 78-81):
  - bed: `data/hg38/human-sequences.bed`
  - fasta: `data/hg38/hg38.ml.fa`
  - `default_data_path` = `$DATA_PATH` or `<repo>/data` (`src/dataloaders/base.py:23-28`).
- `setup()` (lines 94-111) builds the tokenizer and calls `init_datasets()` which creates train/val/test `HG38Dataset` objects, one per split, each with its own `max_length` (`self.max_length`, `max_length_val`, `max_length_test`) (lines 127-142).
- Optional `use_fixed_len_val` swaps the val loader for a fixed, non-overlapping `HG38FixedDataset` over chr14 and chrX (ranges grabbed from Enformer: `chr14:[19726402,106677047]`, `chrX:[2825622,144342320]`, lines 151-162).

**Per-sample dataset:** `HG38Dataset` in `src/dataloaders/datasets/hg38_dataset.py:126-226`.
- Reads the `.bed` file with pandas (columns `chr_name, start, end, split`) and filters to the requested split (lines 164-166).
- `__getitem__` (lines 183-225): takes bed row `idx` -> `(chr, start, end)`, queries the FASTA via `FastaInterval`, tokenizes, and returns `(data, target)` where **`data = seq[:-1]`, `target = seq[1:]`** — the classic **shift-by-one next-token LM** pair (lines 222-224).

**Sequence sampling / interval logic:** `FastaInterval.__call__` in `hg38_dataset.py:72-124`. This is **not** random-interval sampling in the default path — it uses the concrete `(start,end)` from the bed file, then:
- If `interval_length < max_length`: **center-pads** by extending start/end symmetrically to reach `max_length` (lines 95-102).
- If `interval_length > max_length`: **truncates** to `end = start + max_length` (lines 113-114).
- Boundary clamping at chromosome ends, with optional `.`-padding when `pad_interval=True` (lines 104-122).
- Optional augmentations: `shift_augs` (random genomic shift, lines 81-90) and `rc_aug` (reverse-complement coin flip, lines 118-119). Both **off by default** for the pretraining configs.
- FASTA read via `pyfaidx.Fasta` (line 55); chromosome lengths cached in `self.chr_lens` (lines 63-69).

So the pretraining "sampling scheme" is: **iterate the fixed genomic intervals defined in the .bed file** (the Basenji `sequences_human.bed` splits), each interval producing one fixed-`max_length` window (padded or truncated). Randomness across epochs comes from DataLoader `shuffle=True` over bed rows, not from random interval start positions (unless `shift_augs`/`rc_aug` are enabled).

For truly random chromosome/interval sampling (used for "pretrain on your own data"), see `SpeciesDataset` (`src/dataloaders/datasets/species_dataset.py`), which randomly samples a chromosome then a random window — README recommends this path for custom pretraining (README lines 217-227).

### 1.2 Tokenization

**Class:** `CharacterTokenizer` in `src/dataloaders/datasets/hg38_char_tokenizer.py:15-149` (subclasses HuggingFace `PreTrainedTokenizer`). It is a true **single-nucleotide character tokenizer** (`_tokenize` = `list(text)`, line 74).

Instantiation used everywhere (e.g. `genomics.py:100-104`):
```python
CharacterTokenizer(characters=['A','C','G','T','N'],
                   model_max_length=self.max_length + 2,   # +2 for special tokens
                   add_special_tokens=False,
                   padding_side='left')   # causal model -> left pad
```

**Vocabulary (12 tokens):** special tokens occupy ids 0-6, nucleotides start at id 7 (`_vocab_str_to_int`, lines 58-67):

| id | token |
|----|-------|
| 0 | `[CLS]` |
| 1 | `[SEP]` (also used as **eos**) |
| 2 | `[BOS]` |
| 3 | `[MASK]` |
| 4 | `[PAD]` |
| 5 | `[RESERVED]` |
| 6 | `[UNK]` |
| 7 | `A` |
| 8 | `C` |
| 9 | `G` |
| 10 | `T` |
| 11 | `N` |

So `vocab_size` = 12 (hyena configs set `vocab_size: 12`; the older attention `hg38.yaml` uses 11). `pad_vocab_size_multiple: 8` rounds the embedding table up to 16. **EOS is the `[SEP]` token (id 1)**, appended when `add_eos=True` (see `build_inputs_with_special_tokens`, lines 86-94). Padding token id = 4; `replace_N_token=True` optionally rewrites `N` (id 11) to pad so it's ignored in loss (`hg38_dataset.py:218-220`).

### 1.3 Sequence lengths (up to 1M)

- `max_length` is set per experiment config under `dataset.max_length`. Pretraining example `hg38_hyena.yaml` uses `1024` with comments noting `262144, 524288`. The `hg38.yaml` experiment sets `700_000`.
- Tokenizer `model_max_length = max_length + 2` (room for special tokens).
- The Hyena layer's `l_max` is tied to sequence length: `l_max: ${eval:${dataset.max_length}+2}` (`hg38_hyena.yaml`). For downstream configs `l_max` is hard-set (e.g. `1026`) to match the pretrained checkpoint.
- HF released checkpoints support 1k / 16k / 32k / 160k / 450k / 1M (README lines 36-42; `huggingface.py:176-182` map names to max lengths). The 1M model is 8 layers, d_model=256 (`huggingface.py:150-151`) and requires 8xA100 80GB.
- Long sequences are trained via **sequence-length warmup** (see §2.5).

### 1.4 Downstream / fine-tuning datasets

All live in `src/dataloaders/genomics.py` (LightningDataModules) + `src/dataloaders/datasets/*.py` (torch Datasets). Each downstream module subclasses `HG38`.

1. **GenomicBenchmarks** — `GenomicBenchmark` (`genomics.py:218-298`) + `GenomicBenchmarkDataset` (`datasets/genomic_bench_dataset.py:122-209`).
   - Auto-downloads via `genomic_benchmarks.loc2seq.download_dataset` (lines 155-157). Reads plain-text sequence files per class-label folder.
   - 8 datasets configured in `configs/dataset/genomic_benchmark.yaml` (e.g. `human_enhancers_cohn`, `human_nontata_promoters`, `human_ensembl_regulatory` [3 classes], `demo_human_or_worm`, etc.), with per-dataset `train_len` and `classes`.
   - Sequence-level classification, `add_eos=False`, left-padding.

2. **Nucleotide Transformer benchmark** — `NucleotideTransformer` (`genomics.py:301-387`) + `NucleotideTransformerDataset` (`datasets/nucleotide_transformer_dataset.py:26-106`).
   - Reads `.fasta` files (`all_train_enhancer.fasta`, `H3_train.fasta`, etc.) with `pyfaidx`; label = last char of the FASTA header (line 73).
   - 17-18 tasks in `configs/dataset/nucleotide_transformer.yaml` with per-task `max_length`, `classes`, and **`metric`**: `mcc` for enhancer/histone tasks, `f1_macro` for promoter/splice tasks.

3. **Chromatin profile (DeepSEA/BigBird)** — `ChromatinProfile` (`genomics.py:390-461`) + `ChromatinProfileDataset` (`datasets/chromatin_profile_dataset.py:113-270`).
   - `d_output: 919` multilabel targets. Reads coordinates+targets from CSV, queries a reference FASTA (hg19 or hg38). Includes **hg19->hg38 liftover** logic (lines 227-269). Window recentered from 1000bp to `max_length` (lines 176-177).

4. **Species classification** — `Species` (`genomics.py:464-569`) + `SpeciesDataset` (`datasets/species_dataset.py`).
   - Randomly samples chromosome then window from per-species FASTA files; `d_output = len(species)`. Also serves as the template for custom-data pretraining.

5. **ICL / instruction-tuned genomics** — `ICLGenomics` (`genomics.py:572-657`) + `icl_genomics_dataset.py`, `hg38_icl_dataset.py`. Used for in-context / soft-prompt experiments.

---

## 2. TRAINING LOGIC

### 2.1 Entrypoint & framework

**Entrypoint:** `train.py` (Hydra `@hydra.main(config_path="configs", config_name="config.yaml")`, line 679). Launched as `python -m train ...`.

- Framework: **PyTorch Lightning 1.8.6** (pinned in `requirements.txt`) + **Hydra/OmegaConf**.
- Core class: `SequenceLightningModule(pl.LightningModule)` (`train.py:124-573`). It:
  - builds the dataset from `SequenceDataset.registry[...]` (line 138),
  - instantiates model, task, encoder, decoder in `setup()` (lines 150-203),
  - `forward()` delegates to `self.task.forward(batch, encoder, model, decoder, state)` (lines 307-308),
  - `_shared_step` computes loss + metrics + torchmetrics (lines 320-361),
  - `configure_optimizers` builds the optimizer with **per-parameter-group hyperparameters** (params carrying a `_optim` attribute get their own group — used by Hyena filter/pos-emb params) plus the LR scheduler (lines 443-523).
- Custom OmegaConf resolvers: `eval` and `div_up` (lines 37-38). Also a `CustomWandbLogger` with retry logic.

**Config system (Hydra composition):**
- Top: `configs/config.yaml` -> `defaults: experiment: base`.
- `configs/experiment/hg38/*.yaml` are the real entrypoints; each sets `defaults: - /pipeline: <name>` and overrides `model`, `dataset`, `trainer`, `optimizer`, `scheduler`, `task`.
- `configs/pipeline/*.yaml` compose `/trainer + /loader + /dataset + /task + /optimizer + /scheduler + /callbacks` and define `encoder`/`decoder` and the `train.monitor`/`mode`.
- Other groups: `configs/model/` (+ `configs/model/layer/hyena.yaml`), `configs/task/`, `configs/optimizer/`, `configs/scheduler/`, `configs/callbacks/`, `configs/loader/default.yaml`.

### 2.2 Pretraining objective

**Next-token prediction (causal LM), cross-entropy loss.**
- Task config `configs/task/lm.yaml` -> `_name_: lm`, metrics `ppl`. The `hg38` pipeline (`configs/pipeline/hg38.yaml`) sets `task: {_name_: lm, loss: cross_entropy, torchmetrics: ['perplexity','num_tokens']}`.
- `LMTask.forward` (`src/tasks/tasks.py:162-181`): runs encoder->model->decoder, then flattens logits `'... C -> (...) C'` and targets `'... -> (...)'` before loss.
- Loss = `cross_entropy` (`src/tasks/metrics.py:180-183`, `F.cross_entropy` with `ignore_index=-100`; pad token id 4 can be passed as `+task.loss.ignore_index=4` for variable-length data).
- Targets are the input shifted by one (`hg38_dataset.py:222-224`).
- The `HG38Task` variant (`tasks.py:244-329`, `_name_: hg38`) extends `LMTask` to add custom metrics (`last_k_ppl`, `per_token_ppl`).
- For pretraining, `encoder: null` and `decoder: null` (pipeline hg38) — the model's own `lm_head` produces logits directly (see `ConvLMHeadModel`).

### 2.3 Model architecture

**Backbone:** `LMBackbone` / `ConvLMHeadModel` in `src/models/sequence/long_conv_lm.py`.
- `ConvLMHeadModel` (lines 400-502): GPT-2-style stack — `GPT2Embeddings` (token embeddings, **no** positional embeddings for Hyena) -> N x `Block` (prenorm; mixer + MLP with fused dropout/add/layernorm from flash-attn) -> final LayerNorm -> `lm_head` (`nn.Linear(d_model, vocab_size, bias=False)`).
- **Weight tying:** `lm_head.weight = backbone.embeddings.word_embeddings.weight` (`tie_weights`, lines 482-483).
- Blocks are `flash_attn.modules.block.Block`; MLP is GELU (`create_mlp_cls`, lines 102-136); mixer chosen per-layer: attention (`MHA`) if `layer_idx in attn_layer_idx`, else the registry layer (Hyena) (`create_mixer_cls`, lines 48-99).
- `DNAEmbeddingModel` (`src/models/sequence/dna_embedding.py:18-80`) is the same backbone but returns hidden states (no head) for downstream tasks (`_name_: dna_embedding`).

**Hyena operator:** `HyenaOperator` in `src/models/sequence/hyena.py:270-449` (paper arXiv:2302.10866).
- `in_proj` produces `(order+1)*d_model`; a short depthwise `Conv1d` (`short_filter_order=3`) provides local mixing (lines 363-369, 391-394).
- The **implicit long filter** `HyenaFilter` (lines 158-267): an MLP (`emb_dim -> filter_order -> ... -> d_model`) over complex-exponential `PositionalEmbedding` (lines 109-131), with `Sin` activations (freq `w=10`) and `ExponentialModulation` decay (lines 134-155). This parametrizes an arbitrarily long convolution kernel.
- Long convolution done via **FFT** (`fftconv_ref`, lines 59-88, or fused `fftconv_func`). Data-controlled gating multiplies projections then convolves recursively over `order` (lines 414-423).
- Filter params get custom optimizer settings (`lr`, `wd=0`) via the `_optim` attribute (lines 224-227); `lr_pos_emb` controls positional-embedding LR.

**Key hyperparameters (from example configs):**

| Config | d_model | n_layer | d_inner | vocab | layer | l_max | filter_order | emb_dim | max_length | batch_size |
|--------|---------|---------|---------|-------|-------|-------|--------------|---------|------------|------------|
| `hg38_hyena.yaml` (pretrain) | 32 | 2 | 4·d | 12 | hyena | max_len+2 | 64 | 5 | 1024 | 256 |
| `hg38_hyena_seqlen_warmup_reload.yaml` | 32 | 2 | 4·d | 12 | hyena | max_len+2 | 64 | 5 | up to 32768 | 8-256 |
| `genomic_benchmark.yaml` (finetune) | 128 | 2 | 4·d | 12 | hyena | 1026 | 64 | 5 | 256 | 128 |
| `nucleotide_transformer.yaml` (finetune) | 256 | 2 | 4·d | 12 | hyena | 1026 | 64 | 5 | task | 128 |

README's pretrain launch bumps the model to `d_model=128, n_layer=2`. The paper's 1M model = 8 layers, d_model=256. `configs/model/layer/hyena.yaml` holds the default layer config; `short_filter_order=3`, `modulate=True`, `w=10`.

### 2.4 Optimizer, LR schedule, hyperparameters

From `configs/experiment/hg38/hg38_hyena.yaml`:
- **Optimizer:** AdamW (`configs/optimizer/adamw.yaml`), `lr: 6e-4`, `weight_decay: 0.1`, `betas: [0.9, 0.999]`. Hyena filter/pos-emb params overridden to `wd=0`.
- **Scheduler:** `cosine_warmup_timm` (overridden in the experiment). `t_initial` = total steps = `ceil(__train_len / global_batch_size) * max_epochs`; `warmup_t` = 1% of total; `warmup_lr_init: 1e-6`; `lr_min: 0.1*lr`; `t_in_epochs: False` (step-based).
- **Trainer:** `precision: 16` (fp16; bf16 only on A100), `gradient_clip_val: 1.0`, `max_epochs: 100`, `devices: 1`, `accumulate_grad_batches = div_up(global_batch_size, devices*batch_size*num_nodes)`.
- `global_batch_size: 256`, `seed: 2222`.
- Dropout: `embed_dropout: 0.1`, `resid_dropout: 0.0`.

### 2.5 Sequence-length warmup (long-context training)

`configs/experiment/hg38/hg38_hyena_seqlen_warmup_reload.yaml` + callback `seqlen_warmup_reload`. Trains in stages of increasing `seq_len` (1024 -> 2048 -> 4096 -> 8192 -> 16384 -> 32768) with decreasing `batch_size` (256 -> 8), keeping `global_batch_size=256` by adjusting grad-accumulation. `train.py:625-642` builds the `accumulate_grad_batches` schedule and forces `DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)`.

### 2.6 Launch commands (from README)

Download hg38 data first (README lines 197-199):
```bash
mkdir -p data/hg38/
curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz > data/hg38/hg38.ml.fa.gz
# gunzip it -> data/hg38/hg38.ml.fa
curl https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed > data/hg38/human-sequences.bed
```
Expected layout: `data/hg38/hg38.ml.fa` + `data/hg38/human-sequences.bed`.

Pretrain (README line 205):
```bash
python -m train wandb=null experiment=hg38/hg38_hyena \
  model.d_model=128 model.n_layer=2 \
  dataset.batch_size=256 train.global_batch_size=256 \
  dataset.max_length=1024 optimizer.lr=6e-4 trainer.devices=1
```

---

## 3. EVALUATION LOGIC

### 3.1 Loading pretrained checkpoints

Two paths:

**(A) In-repo PyTorch Lightning finetuning** (README lines 154-170):
- Pass `train.pretrained_model_path=/path/to/ckpt`. `train.py:656-662` calls `SequenceLightningModule.load_from_checkpoint(..., strict=config.train.pretrained_model_strict_load)`.
- A state-dict hook `load_backbone` (configs set `train.pretrained_model_state_hook: {_name_: load_backbone, freeze_backbone: false}`) rewrites the checkpoint so the **backbone loads but the head/decoder are trained from scratch** (`src/models/sequence/long_conv_lm.py:569-627` and `dna_embedding.py:83-142`). `pretrained_model_strict_load: False` lets the new encoder/decoder be added.

**(B) Standalone / HuggingFace** (`huggingface.py`, `standalone_hyenadna.py`):
- `HyenaDNAPreTrainedModel.from_pretrained(path, model_name, download=True, ...)` (`huggingface.py:70-126`) git-clones `https://huggingface.co/LongSafari/{model_name}`, reads `config.json` + `weights.ckpt`, and performs state-dict "surgery" via `load_weights`/`inject_substring` (handles the `.model` prefix and the extra `.layer` inserted by gradient checkpointing) (lines 27-68, 108-124).
- `standalone_hyenadna.py` is a **fully self-contained** model definition (`HyenaDNAModel`, its own `HyenaOperator`, `CharacterTokenizer`, `SequenceDecoder`) for colab/inference with no repo dependencies.
- `evals/hg38_inference.py` (`HG38Encoder`) loads a `ConvLMHeadModel` from a yaml `model_cfg` + `ckpt_path`, strips `model.`/`torchmetrics.` keys, and runs inference to produce logits/embeddings.

### 3.2 Pretraining metric (PPL)

- Loss metric `ppl(x, y, loss_fn) = exp(loss_fn(x,y))` (`src/tasks/metrics.py:310-311`), registered in `loss_metric_fns` (lines 345-349).
- Also `torchmetrics: ['perplexity','num_tokens']` in the hg38 pipeline (streaming torchmetrics defined in `src/tasks/torchmetrics.py`).
- `HG38Task` adds `last_k_ppl` (PPL over final k tokens, `metrics.py:89-118`) and `per_token_ppl` (`ppl_at_{k}`) to measure long-range recall (`tasks.py:280-329`).
- `train.monitor: test/loss`, `mode: min` for pretraining (`pipeline/hg38.yaml`).

### 3.3 Downstream benchmark evaluation

- **Head:** downstream models use `_name_: dna_embedding` (returns hidden states); a `SequenceDecoder` (`src/tasks/decoders.py:38-142`) with `mode: pool` (mean over tokens) or `mode: last`, `l_output: 0` (single sequence-level output) maps to `d_output` classes. Pipeline sets `encoder: id`, `decoder: {_name_: sequence, mode: pool}` (`pipeline/genomic_benchmark.yaml`, `pipeline/nucleotide_transformer.yaml`).
- **Task:** `masked_multiclass` (`MaskedMultiClass`, `tasks.py:224-241`) — cross-entropy classification that can honor an attention mask so only real tokens are pooled.
- **Metrics** (`src/tasks/metrics.py`):
  - GenomicBenchmarks: `accuracy` (`metrics.py:192-199`), monitor `val/accuracy` (max), `plateau` scheduler.
  - Nucleotide Transformer: metric selected per-task from config — **`mcc`** (`matthews_corrcoef`, lines 82-86) or **`f1_macro`** (`f1_score(..., average='macro')`, lines 227-231); monitor `val/${dataset.metric}` (max).
  - Chromatin profile: multilabel -> `binary_cross_entropy` + torchmetrics AUROC/Precision/Recall/F1 (`configs/task/multilabel_classification.yaml`).
- The README notes a **prebuilt Docker image** (`launch_commands_nucleotide_transformer`) with the exact datasets/splits/hyperparameters to reproduce the paper's NT results (README lines 122-135, 267).

**Downstream launch commands (README):**
```bash
# GenomicBenchmarks (finetune from pretrained ckpt)
python -m train wandb=null experiment=hg38/genomic_benchmark \
  dataset_name=human_enhancers_cohn \
  train.pretrained_model_path=/path/to/ckpt \
  dataset.max_length=500 model.layer.l_max=1024

# GenomicBenchmarks from scratch (auto-downloads data)
python -m train wandb=null experiment=hg38/genomic_benchmark_scratch

# Nucleotide Transformer
python -m train wandb=null experiment=hg38/nucleotide_transformer \
  dataset_name=enhancer dataset.max_length=500 model.layer.l_max=1026

# Chromatin profile
python -m train wandb=null experiment=hg38/chromatin_profile \
  dataset.ref_genome_path=/path/to/hg38.ml.fa \
  dataset.data_path=/path/to/chromatin_profile dataset.ref_genome_version=hg38

# Species classification
python -m train wandb=null experiment=hg38/species dataset.species=[human,mouse,hippo,pig,lemur] ...
```
Critical: `model.layer.l_max` on downstream must equal the pretrained model's `l_max` (`dataset.max_length + 2`), or the positional-embedding-derived filter won't match.

---

## 4. REPRODUCTION-RELEVANT NOTES

### 4.1 Dependencies / environment
- Python 3.8+, **PyTorch 1.13.0 + CUDA 11.7** (README). `requirements.txt` pins `pytorch-lightning==1.8.6`, `transformers==4.26.1`, `torchtext==0.14.0`, `hydra-core`, `omegaconf`, `einops`, `wandb`, plus genomics-specific: `pyfaidx`, `polars`, `genomic-benchmarks`, `liftover`.
- **flash-attention** is a git submodule (`.gitmodules`) and a hard dependency of the model code: `long_conv_lm.py` imports `flash_attn.modules.{mha,mlp,block,embedding}` and `flash_attn.utils.generation`. Must clone with `--recurse-submodules` and build (`cd flash-attention && pip install . --no-build-isolation`, plus `csrc/layer_norm`). This is the **hardest install** (needs matching CUDA toolkit, long compile, a GPU). The `Dockerfile` (base `nvcr.io/nvidia/pytorch:22.07-py3`) automates it.
- Optional fused kernels: `fused_dense_lib` (ColumnParallelLinear), `dropout_add_layer_norm`, and the custom FFT conv in `csrc/fftconv` — all guarded by try/except, so partial installs degrade gracefully (fall back to `fftconv_ref`).
- The **standalone_hyenadna.py** path deliberately avoids flash-attn for inference-only use (colab).

### 4.2 Data to download & sizes
- **hg38 fasta** `hg38.ml.fa` (~3GB uncompressed; download is `hg38.ml.fa.gz`) + `human-sequences.bed` (Basenji intervals with train/valid/test split column). From `storage.googleapis.com/basenji_barnyard2/`.
- GenomicBenchmarks and (some) NT datasets auto-download into `data/`. Chromatin profile requires manual DeepSEA/Sei download + hg19/hg38 FASTA. Species requires per-species FASTA zips.
- HF pretrained weights (`LongSafari/hyenadna-*`) via git-lfs.

### 4.3 Gotchas
- **vocab_size mismatch**: the char tokenizer has 12 tokens, but the attention `hg38.yaml` example sets `vocab_size: 11` while hyena configs use `12`. `pad_vocab_size_multiple: 8` pads the embedding to 16 regardless — keep this consistent between pretrain and finetune or checkpoint loading breaks.
- **`l_max` must match** the pretrained model exactly on downstream (`dataset.max_length + 2`), else Hyena positional embeddings mismatch. README lines 238, 272 hard-set it.
- **Gradient-checkpointing flags** (`model.checkpoint_mixer`, `model.checkpoint_mlp`) change parameter key names (extra `.layer`); must match between train and load or you get `Missing key ... mixer.layer.filter_fn.bias` errors (README lines 435-446; handled by `inject_substring` in `huggingface.py`).
- **Left padding** is required (causal model): tokenizer `padding_side='left'`. Fixed in downstream dataset configs.
- **`N` handling**: `N` is a real vocab token (id 11); optionally rewritten to pad (`replace_N_token`) so it can be ignored in loss. For custom data with lots of padding/unknowns, pass `+task.loss.ignore_index=4`.
- **fp16 vs bf16**: configs default `precision: 16` (fp16) with comments that bf16 is A100-only. Long-context stability may need care.
- **"epoch" is synthetic**: `__train_len = 1e9 / max_length`, so scheduler total-step math depends on `max_length` and `global_batch_size`; changing one without the other skews the LR schedule.
- **Deterministic interval sampling**: default hg38 pretraining reads fixed bed intervals (not random crops); reverse-complement/shift augs are off by default. To match "random sampling" narratives, enable `rc_aug`/`shift_augs` or use the species dataloader.
- **Val vs test monitoring**: with `use_fixed_len_val`, the fixed chr14/chrX set is placed in the *val* loader while true tracking uses `test/loss` — read `init_datasets` carefully.

---

## Key file reference

| Concern | File |
|---|---|
| Training entrypoint / Lightning module | `train.py` |
| hg38 datamodule + all downstream datamodules | `src/dataloaders/genomics.py` |
| hg38 per-sample dataset + FASTA interval logic | `src/dataloaders/datasets/hg38_dataset.py` |
| Char tokenizer / vocab | `src/dataloaders/datasets/hg38_char_tokenizer.py` |
| Downstream datasets | `src/dataloaders/datasets/{genomic_bench,nucleotide_transformer,chromatin_profile,species}_dataset.py` |
| LM backbone + head + backbone-load hook | `src/models/sequence/long_conv_lm.py` |
| Embedding-only backbone (downstream) | `src/models/sequence/dna_embedding.py` |
| Hyena operator | `src/models/sequence/hyena.py` |
| Tasks (LM, HG38, MaskedMultiClass) | `src/tasks/tasks.py` |
| Losses & metrics (CE, ppl, mcc, f1) | `src/tasks/metrics.py` |
| Decoder heads (pooling) | `src/tasks/decoders.py` |
| Configs | `configs/{config.yaml,experiment/hg38/*,pipeline/*,dataset/*,task/*,model/layer/hyena.yaml,optimizer/*,scheduler/*}` |
| HF/standalone loading & inference | `huggingface.py`, `standalone_hyenadna.py`, `evals/hg38_inference.py` |
| Env / deps | `requirements.txt`, `Dockerfile`, `.gitmodules` |
