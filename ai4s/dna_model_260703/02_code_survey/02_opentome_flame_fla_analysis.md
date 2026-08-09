# OpenToMe / flame / flash-linear-attention — Code Survey for DNA Pretraining Reproduction

Date: 2026-07-03
Repos surveyed:
- `/Users/lisiyuan/Downloads/local_exp/_claude_discussion/dna_model_260703/02_code_survey/repos/OpenToMe`
- `/Users/lisiyuan/Downloads/local_exp/_claude_discussion/dna_model_260703/02_code_survey/repos/flame`
- `/Users/lisiyuan/Downloads/local_exp/_claude_discussion/dna_model_260703/02_code_survey/repos/flash-linear-attention`
- `/Users/lisiyuan/Downloads/local_exp/_claude_discussion/dna_model_260703/02_code_survey/repos/hnet` (also present)

---

## 0. TL;DR — Correcting the assumption

**OpenToMe is, by its own README title, a "Toolbox and Benchmark for Token Merging Modules"** — i.e. primarily a *vision-transformer token-merging / token-compression* research toolbox (ToMe, DiffRate, ToFu, PiToMe, DTEM, etc.), used for ImageNet classification, ToMe visualization, and throughput benchmarks. See `OpenToMe/README.md:1` and `OpenToMe/setup.py:107` (`description='Open-source Toolbox for Token Merging Modules'`).

**HOWEVER**, this particular checkout of OpenToMe has been *extended by a contributor ("jinxin")* into a **language-model pretraining superset**. It now:
- Vendors a **full modified copy of `flame`** at `OpenToMe/trainer/flame/` (its own `train.py`, `data.py`, configs, and shell scripts).
- Ships **FLA-style LM model definitions** under `OpenToMe/opentome/models/` (`transformer`, `delta_net`, `gated_deltanet`, `gla`, `gsa`, `blt`, `qwen3_next`, plus a custom `mergenet`/`mergenet_nlp`).
- Provides **byte-level pretraining scripts** at `OpenToMe/trainer/flame/scripts/byte/` and `byte_1b/` — directly relevant to single-nucleotide / byte DNA modeling.

So the user's belief is **partially correct**: OpenToMe as published upstream is a vision token-merging toolbox, **but this working copy is genuinely wired for FLA language-model pretraining via a vendored flame**. It is *not* built "on top of" fla+flame in a clean dependency sense — rather it **bundles a patched flame and re-implements the fla model classes locally**.

**The real training engine is `flame` (which is built on `torchtitan`), with `fla` (flash-linear-attention) supplying the model architectures.** OpenToMe here is a wrapper/superset that adds: custom tokenizers (byte/BLT/sentencepiece/tiktoken), custom optimizers (Muon/SCALE/RMNP), PPL validation during training, and local model copies. For the DNA reproduction, **the operative code is `OpenToMe/trainer/flame/`, not the vision toolbox.**

---

## 1. OpenToMe

### 1.1 Actual purpose
- README title (`OpenToMe/README.md:1`): *"Toolbox and Benchmark for Token Merging Modules"*. News entry dates the framework to 2025-06-23.
- `setup.py`: `name='OpenToMe'`, `keywords='efficient transformer, token merging'`, author "CAIRI Westlake University".
- Baselines table lists vision token-compression methods (ToMe ICLR'23 … FPET CVPR'25).
- "Support Tasks" checklist: Image Classification, LLM Inference, **Long Sequence Training**, Throughput, ToMe Visualization, Optimizers — i.e. the LM-training capability is an explicitly-added feature, not the original core.

### 1.2 Directory structure
```
OpenToMe/
├── opentome/                # the installable package (vision + NLP)
│   ├── tome/                # token-merging algorithms (ToMe, PiToMe, DTEM, ...)
│   ├── timm/                # timm ViT blocks patched for ToMe
│   ├── models/              # BOTH vision (deit, mergenet) AND FLA LM copies
│   │   ├── transformer/     #   Transformer++ (LLaMA-like) — copy of fla
│   │   ├── delta_net/       #   DeltaNet — copy of fla
│   │   ├── gated_deltanet/  gla/ gsa/ blt/ qwen3_next/ mergenet_nlp/
│   ├── tokenizer/           # byte/BLT/sentencepiece/tiktoken tokenizers
│   ├── optimizer/           # custom optimizers (Muon etc.)
│   └── utils/               # optimization.py (build_optimizers for SCALE/RMNP/Muon)
├── trainer/
│   ├── classification/      # ViT ImageNet/CIFAR training (vision core)
│   └── flame/               # ← VENDORED, PATCHED flame (the LM training engine)
├── evaluations/             # image classification, lm_harness, visualizations
├── demo/, docs/, test/
├── setup.py, requirements/, fla_environment.yml
```

Main entrypoints:
- **Vision:** `trainer/classification/train.py`, `evaluations/image_classification/in1k_example.py`.
- **LM pretraining (relevant):** `trainer/flame/train.sh` → `python -m flame.train` (`trainer/flame/flame/train.py`).

### 1.3 Dependency on fla + flame — evidence
- `OpenToMe/README.md:27-28` "Install FLA (flash-linear-attention) … install the requested environment for training."
- `OpenToMe/README.md:139-176` "Flash Linear Attention Model Training … example of training with flash linear attention by **flame**".
- `OpenToMe/fla_environment.yml` pins the real stack: `fla-core==0.4.0`, `flame==0.1.0`, `flash-linear-attention==0.4.0`, `torchtitan==0.1.0`, `flash-attn==2.7.4.post1`, `torch==2.6.0`, `transformers==4.57.3`, `triton==3.2.0`, `datasets==4.4.1`, `torchdata==0.11.0`.
- The local LM model files import fla directly, e.g. `opentome/models/delta_net/modeling_delta_net.py` imports `from fla.layers.delta_net import DeltaNet`, `from fla.modules import ...`, `from fla.layers.attn import Attention`.
- `trainer/flame/flame/train.py:21` `import fla`, and `:23` `from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss`; heavy `torchtitan.*` imports throughout.
- NOTE: the base `requirements/runtime.txt` (einops, fvcore, lpips, timm, scikit-learn…) is **vision-only** — fla/flame/torchtitan are *not* in it; they come from `fla_environment.yml`. This confirms LM training is a bolt-on that requires the separate `fla` conda env.

### 1.4 Transformer++ / DeltaNet model definitions
Yes — present in **two** places:
- `opentome/models/transformer/{configuration_transformer.py,modeling_transformer.py}` — `TransformerConfig(model_type='transformer')`, `TransformerForCausalLM`. This is the fla "Transformer++" (LLaMA-like: RoPE, SwiGLU, GQA, RMSNorm).
- `opentome/models/delta_net/{configuration_delta_net.py,modeling_delta_net.py}` — `DeltaNetConfig(model_type='delta_net')`, `DeltaNetForCausalLM`.
- Each `__init__.py` registers with HF Auto classes, e.g. `opentome/models/delta_net/__init__.py:7-9`:
  ```python
  AutoConfig.register(DeltaNetConfig.model_type, DeltaNetConfig, exist_ok=True)
  AutoModel.register(DeltaNetConfig, DeltaNetModel, exist_ok=True)
  AutoModelForCausalLM.register(DeltaNetConfig, DeltaNetForCausalLM, exist_ok=True)
  ```
- These are **thin local copies/edits of the fla originals** (OpenToMe delta_net modeling = 369 lines vs fla's 455; both import `fla.layers.delta_net.DeltaNet` for the actual kernel). They exist so OpenToMe can patch/extend them and so `train.py` can `import opentome.models.<x>` to trigger Auto-registration.

### 1.5 Pretraining capability
**Yes — full from-scratch causal-LM pretraining**, via the vendored flame (see §2). It is *not* vision-only and *not* inference-only. The `trainer/flame/scripts/` tree contains ready 340M and 1B recipes for transformer++, deltanet, gated_deltanet, gla, gsa, blt, mergenet, qwen3_next — and crucially a **`byte/` set** for byte-level (256-vocab) pretraining.

---

## 2. flame (both the upstream `repos/flame` and the vendored `OpenToMe/trainer/flame`)

### 2.1 What it is — confirmed
`flame/README.md:9`: *"🔥 `flame`, a minimal and efficient framework built on `torchtitan` for language models."* Feature highlights: minimal extensible training framework; seamless `fla` + `transformers` integration; zero-cost online tokenization, dataset shuffling, multi-dataset; 4D parallelism (coming). **Confirmed: flame = thin training harness on torchtitan; fla = models.**

### 2.2 Training entrypoint & config system
- Launch script: `train.sh` → `torchrun … -m flame.train "${params[@]}"`.
- `flame/train.py` uses torchtitan's `JobConfig` (`flame/config_manager.py`) — a **hybrid TOML + CLI** system:
  - A base **TOML** job config (`flame/models/fla.toml`) sets defaults (`[model]`, `[training]`, `[optimizer]`, `[lr_scheduler]`, `[checkpoint]`, `[metrics]`, `[experimental]`, `[float8]`, `[activation_checkpoint]`).
  - CLI flags (`--training.seq_len`, `--model.config`, …) override TOML.
- torchtitan integration: `register_train_spec(TrainSpec(name="fla", cls=AutoModelForCausalLM, config=AutoConfig, parallelize_fn=parallelize_fla, pipelining_fn=pipeline_fla, build_dataloader_fn=build_dataloader, build_tokenizer_fn=build_tokenizer, build_loss_fn=build_cross_entropy_loss))` (`trainer/flame/flame/train.py:284`). Model instantiated on `meta` device then sharded: `AutoModelForCausalLM.from_config(model_config)` (`train.py:439`).

### 2.3 Data pipeline (`flame/data.py`)
- Uses **HuggingFace `datasets`** (`load_dataset`) with optional **streaming**. `build_dataset()` (`data.py:555`) supports a single dataset or comma-separated multiple datasets interleaved by `--training.data_probs`.
- Expected format: a HF dataset (local dir, HF hub id, or arrow/parquet loadable by `datasets`) with a **`"text"` or `"content"` column** — see `OnlineTokenizedIterableDataset.tokenize` (`data.py:198-203`), which raises if neither field exists.
- **Online tokenization** is the default: raw text → tokenizer → token stream, packed into `seq_len` chunks by `OnlineTokenizedIterableDataset` (`data.py:156`). A `BufferShuffledIterableDataset` variant exists for shuffled streaming.
- Collation: `DataCollatorForLanguageModeling` (`data.py:321`) supports padded batches (`varlen=False`) and **packed variable-length** (`varlen=True`, batch size must be 1) — the latter builds `cu_seqlens` from BOS/EOS positions and splits at `context_len`. Labels = input_ids clone (`-100` on pad).
- There is a `flame/utils/preprocess.py` for optional offline preprocessing, but the primary path is **on-the-fly tokenization of a text/parquet HF dataset** (no pre-tokenized requirement).
- The OpenToMe-vendored `train.py` also adds a **PPL validation** path (`build_val_chunks_cache`, `evaluate_ppl`, `train.py:96-281`) reading a wiki_val parquet or C4 json.gz.

### 2.4 Model selection
- Selected by the **`--model.config <path.json>`** whose `"model_type"` field routes HF `AutoConfig`/`AutoModelForCausalLM` to the right registered class. Configs live in `flame/configs/` (upstream) and `OpenToMe/trainer/flame/configs/` (e.g. `delta_net_340M.json` has `"model_type": "delta_net"`; `transformer_340M.json` has `"model_type": "transformer"`).
- **OpenToMe-specific twist:** the vendored `train.py:43-70` reads env var **`BACKBONE`** and does `import opentome.models.<backbone>` to trigger Auto-registration of the *local* model copies before `AutoModelForCausalLM.from_config`. So on OpenToMe you must `export BACKBONE=delta_net_340M` (etc.), as the byte/350m scripts do.
- Upstream flame instead relies on `import fla` auto-registering all fla model types.

### 2.5 Tokenizer handling — custom DNA tokenizer feasibility
- Default (`train.py:383-389`): `AutoTokenizer.from_pretrained(job_config.model.tokenizer_path, trust_remote_code=True)` — assumes an HF tokenizer directory/hub id.
- **OpenToMe adds a non-HF path** (`train.py:390-401`) gated by env `TOKENIZER_NAME ∈ {bytes, sentencepiece, tiktoken, blt}`, building via `opentome.tokenizer.build_tokenizer.TokenizerArgs`. The byte/BLT tokenizer uses `vocab_size_unit_1=256` and sets `model_config.vocab_size = tokenizer.get_vocab_size()` (`train.py:433`).
- **DNA implication:** a single-nucleotide tokenizer (A/C/G/T/N → ~5–8 ids, or a byte tokenizer) is fully supported. Two options:
  1. Wrap a tiny HF `PreTrainedTokenizerFast`/`PreTrainedTokenizer` (vocab of nucleotides + BOS/EOS) and point `--model.tokenizer_path` at it (default path). This is the cleanest — the data collator needs `bos_token_id`/`eos_token_id` for varlen packing.
  2. Reuse the built-in **byte tokenizer** (`TOKENIZER_NAME=bytes` or `blt`, 256 vocab) and encode DNA as ASCII bytes — matches the existing `scripts/byte/*` recipes.
- The dtype logic in `data.py:44-49` auto-selects `uint16`/`uint32` from `tokenizer.vocab_size`, so a tiny DNA vocab is fine.

### 2.6 Distributed features (via torchtitan)
- **FSDP / HSDP / DDP**: `--training.data_parallel_shard_degree`, `--training.data_parallel_replicate_degree` (`ParallelDims`, `train.py:328`).
- **Tensor Parallel**: `--training.tensor_parallel_degree` (with `--training.disable_loss_parallel`).
- **Context Parallel**: `--experimental.context_parallel_degree` (`create_context_parallel_ctx`, `train.py:789`).
- **Pipeline Parallel**: scaffolding present but **explicitly `NotImplementedError`** in this version (`train.py:363,803`).
- Also: `torch.compile` (`--training.compile`), CPU offload, float8, activation checkpointing, async DCP checkpointing, `--training.varlen` packing.

### 2.7 Exact launch command (DeltaNet 340M byte-level, from `OpenToMe/trainer/flame/scripts/byte/deltanet.sh`)
```bash
export HF_ENDPOINT=https://hf-mirror.com
export BACKBONE=delta_net_340M        # triggers import opentome.models.delta_net
export TOKENIZER_NAME=blt             # byte/BLT tokenizer (256 vocab)

NNODE=1 NGPU=4 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/byte/delta_net_340M_10B/... \
  --model.config configs/delta_net_340M.json \
  --model.tokenizer_path /path/to/tokenizer \
  --optimizer.name AdamW --optimizer.lr 3e-4 --optimizer.eps 1e-15 \
  --lr_scheduler.warmup_steps 1024 --lr_scheduler.lr_min 0.1 --lr_scheduler.decay_type cosine \
  --training.batch_size 1 --training.seq_len 32768 --training.context_len 4096 --training.varlen \
  --training.gradient_accumulation_steps 4 --training.steps 30720 --training.max_norm 1.0 --training.skip_nan_inf \
  --training.dataset /path/to/hf_dataset --training.dataset_name default --training.dataset_split train \
  --training.num_workers 32 --training.prefetch_factor 2 --training.seed 42 --training.compile \
  --checkpoint.interval 15360 --checkpoint.load_step -1 --checkpoint.keep_latest_k 2 --metrics.log_freq 1
```
`train.sh` finishes by auto-converting DCP → HF format via `flame.utils.convert_dcp_to_hf`.

---

## 3. flash-linear-attention (fla)

### 3.1 Available models under `fla/models/`
Confirmed present (all requested + many more): `transformer` (Transformer++), `delta_net`, `gated_deltanet`, `mamba`, `mamba2`, `mamba3`, `hgrn`, `hgrn2`, `gla`, `retnet`, plus: `abc`, `bitnet`, `comba`, `deltaformer`, `forgetting_transformer`, `gated_deltaproduct`, `gsa`, `kda`, `lightnet`, `linear_attn`, `log_linear_mamba2`, `mesa_net`, `mla`, `moba`, `mom`, `nsa`, `parallax`, `path_attn`, `raven`, `rodimus`, `rwkv6`, `rwkv7`, `samba`, `wall_transformer`, `yoco`. All exported/registered in `fla/models/__init__.py`.

### 3.2 Config & ForCausalLM class locations
- **Transformer (Transformer++):**
  - Config: `fla/models/transformer/configuration_transformer.py` → `TransformerConfig` (`model_type='transformer'`). LLaMA-like markers confirmed: `num_kv_heads` (GQA), `qkv_bias`, `window_size`, `rope_theta=10000`, `hidden_act='swish'`, `fuse_swiglu=True`, RMSNorm. This is the standard "Transformer++" baseline.
  - Modeling: `fla/models/transformer/modeling_transformer.py` → `TransformerPreTrainedModel:172`, `TransformerModel:227`, `TransformerForCausalLM:346`.
- **DeltaNet:**
  - Config: `fla/models/delta_net/configuration_delta_net.py` → `DeltaNetConfig` (`model_type='delta_net'`).
  - Modeling: `fla/models/delta_net/modeling_delta_net.py` → `DeltaNetPreTrainedModel:175`, `DeltaNetModel:233`, `DeltaNetForCausalLM:342`.

### 3.3 HuggingFace convention compliance
**Yes.** All fla models subclass `transformers.PreTrainedModel` and mix in `FLAGenerationMixin`, define `config_class`, and are registered via `AutoConfig/AutoModel/AutoModelForCausalLM.register(...)` in each package `__init__.py`. This is exactly why flame can do `AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(json))` generically.

### 3.4 HNet in fla?
**No.** There is no HNet/H-Net implementation anywhere in `fla/` (grep across the repo finds none). HNet exists only as a **separate standalone repo** at `repos/hnet` (see §4).

---

## 4. HNet (`repos/hnet`, goombalab/hnet — "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", arXiv 2507.07955)

- Structure: `hnet/models/{config_hnet.py, hnet.py, mixer_seq.py}`, `hnet/modules/{dc.py (dynamic chunking), isotropic.py, block.py, mha.py, mlp.py, rotary.py}`, `hnet/utils/tokenizers.py`, `generate.py`.
- **NOT HuggingFace-native.** `HNetForCausalLM` (`hnet/models/mixer_seq.py:22`) subclasses `nn.Module` + **`flash_attn.utils.generation.GenerationMixin`** (not `PreTrainedModel`), and config is a plain `@dataclass HNetConfig` (`config_hnet.py`), **not** a `PretrainedConfig`. Weights are loaded from raw `.pt` via `generate.py`, not `from_pretrained`.
- **Hard dependencies:** `mamba_ssm` (`hnet/modules/dc.py:9` `mamba_chunk_scan_combined`), `causal_conv1d`, and `flash_attn==2.8.0.post2` (`hnet/pyproject.toml`). These are CUDA/Triton kernels requiring a GPU build.
- Ships a **`ByteTokenizer`** (`hnet/utils/tokenizers.py`, vocab 256, bos=254, eos=255) — HNet is natively byte-level, which aligns well with single-nucleotide/byte DNA modeling.
- Config uses hierarchical `arch_layout`, per-stage `d_model`/`d_intermediate`, `ssm_cfg`, `attn_cfg` — a nested multi-stage architecture unlike the flat fla configs.

---

## 5. MIGRATION MAP — reproducing Transformer++ / DeltaNet / HNet DNA pretraining in OpenToMe

### 5.1 Real relationship (definitive)
- **flame = training engine** (built on torchtitan): data loading, tokenization, FSDP/TP/CP, optimizer, checkpoint, loss loop.
- **fla = model library** (HF-compatible `PreTrainedModel`s): supplies Transformer++, DeltaNet, etc.
- **OpenToMe (this checkout) = superset/wrapper** that vendors a *patched* flame under `trainer/flame/`, re-hosts local copies of the fla model classes under `opentome/models/`, and adds custom tokenizers, optimizers, byte-level recipes, and PPL validation. For DNA work, treat **`OpenToMe/trainer/flame/` as your working root** (it is where the byte scripts, patched `train.py`, and the `BACKBONE`/`TOKENIZER_NAME` env hooks live).

### 5.2 Steps to pretrain an fla model (e.g. DeltaNet) on single-nucleotide hg38 DNA

1. **Environment.** Create the `fla` conda env from `OpenToMe/fla_environment.yml` (torch 2.6, fla 0.4.0, torchtitan 0.1.0, flash-attn 2.7.4, triton 3.2, transformers 4.57.3, datasets 4.4.1). GPU required (Triton kernels). `pip install -e OpenToMe` so `opentome.*` is importable.

2. **DNA tokenizer.** Choose one:
   - *Recommended:* build a minimal HF tokenizer (`PreTrainedTokenizerFast`) with a fixed nucleotide vocab (`A,C,G,T,N` + `<bos>`,`<eos>`,`<pad>`; ~8 tokens), save to a dir, point `--model.tokenizer_path` at it, and leave `TOKENIZER_NAME=default`. Ensure `bos_token_id`/`eos_token_id` are set (required by the varlen collator to build `cu_seqlens`, `data.py:414-467`).
   - *Alternative:* reuse the built-in byte/BLT tokenizer (`export TOKENIZER_NAME=bytes` or `blt`, 256 vocab) and feed DNA as ASCII characters — matches existing `scripts/byte/*.sh`. `model_config.vocab_size` is then auto-set to the tokenizer size.

3. **Dataset prep.** Convert hg38 to an HF dataset with a **`text` column** (each row a sequence/contig or fixed window of nucleotide characters). Save as parquet/arrow (`datasets.save_to_disk` or a parquet dir loadable by `load_dataset`). Point `--training.dataset /path`, `--training.dataset_name default`, `--training.dataset_split train`. No pre-tokenization needed — tokenization is online. For genomic long-context, use `--training.varlen --training.seq_len <packed> --training.context_len <max_contig>`.
   - Provide a validation parquet with a `text` column for the built-in PPL eval, or set `--training.val_data_dir` (see `train.py:651-676`); default expects `./data/wiki_val/validation-00000-of-00001.parquet`, so override it for DNA.

4. **Model config.** Copy `configs/delta_net_340M.json` (or `transformer_340M.json` for Transformer++). Keep `model_type` = `delta_net` / `transformer`. Adjust `hidden_size`, `num_hidden_layers`, `num_heads`; `vocab_size` will be overridden from the tokenizer at runtime (`train.py:430/433`).

5. **Launch.** Use `scripts/byte/deltanet.sh` (or `scripts/350m/deltanet.sh`) as the template: set `export BACKBONE=delta_net_340M` (must contain the substring the dispatcher matches — `delta_net`, `transformer++`, etc., `train.py:45-68`) and the DNA tokenizer/dataset paths, then run the `bash train.sh …` command in §2.7. For a single-GPU smoke test set `NGPU=1`.

6. **Checkpoint → HF.** `train.sh` auto-runs `flame.utils.convert_dcp_to_hf` at the end for downstream eval (lm-eval-harness path in README).

**For Transformer++:** identical flow with `--model.config configs/transformer_340M.json` and `export BACKBONE=transformer++_340M` (the dispatcher matches `transformer++` → `import opentome.models.transformer`). Confirmed recipe: `scripts/byte/transformer++.sh`.

### 5.3 Adding HNet (not in fla) — what it takes
HNet is **not** a simple config registration; it is a foreign, non-HF model. Two integration routes:

- **Route A — HF-wrap into flame (preferred for reuse of the flame engine).** Follow flame's "Custom models" recipe (`flame/README.md:487-499`, template `flame/custom_models/sba/`):
  1. Create `custom_models/hnet/` with `config_hnet.py` defining an `HNetConfig(PretrainedConfig)` (wrap/serialize the existing dataclass fields: `arch_layout`, `d_model[]`, `d_intermediate[]`, `ssm_cfg`, `attn_cfg`, `vocab_size`).
  2. Wrap `hnet.models.mixer_seq.HNetForCausalLM` in a `PreTrainedModel` subclass exposing the flame-expected forward signature `(input_ids, labels, position_ids, cu_seqlens) -> output.loss` and a `post_init()`/`_is_hf_initialized` path (flame builds on `meta` device then calls `post_init`, `train.py:438-506`).
  3. Register with `AutoConfig`/`AutoModelForCausalLM` in `__init__.py`; add `configs/hnet_*.json` with the new `model_type`; add the `import` branch in the OpenToMe `train.py` `BACKBONE` dispatcher (or use flame's `--experimental.custom_model_path`).
  - HNet is natively byte-level (256 vocab), so the byte/DNA tokenizer path fits directly.

- **Route B — use HNet's own training code.** `hnet/utils/train.py` + `generate.py` exist, but the repo ships **inference/generation** primarily; a full pretraining loop comparable to flame's (FSDP, checkpoint resume, data streaming) is not provided, so you'd be building the harness yourself.

### 5.4 What makes this hard (risks / blockers)
1. **HNet is not HF-native** (`nn.Module` + flash_attn `GenerationMixin`, dataclass config, `.pt` weights). Meaningful adapter work is required to run it under flame; the `post_init`/meta-device init and DTensor sharding (`train.py:494-506`) assume HF `PreTrainedModel` semantics.
2. **Heavy CUDA kernel deps for HNet:** `mamba_ssm`, `causal_conv1d`, `flash_attn==2.8.0.post2` — must be compiled for the target GPU/CUDA; version-sensitive and can conflict with the fla env's `flash-attn==2.7.4`.
3. **HNet's dynamic chunking + hierarchical `arch_layout`** complicate FSDP/TP sharding and `cu_seqlens`/`position_ids` handling that flame feeds models (`train.py:774-799`); TP/CP correctness for HNet is unproven here.
4. **Pipeline parallelism is disabled** (`NotImplementedError`, `train.py:363,803`) — only FSDP/HSDP/DDP + TP + CP are usable.
5. **OpenToMe vendors *modified copies* of both flame and the fla models**, which can drift from upstream fla/flame; ensure `import fla` resolves to the pinned `fla==0.4.0` and that the local `opentome.models.*` registration (via `BACKBONE`) doesn't clash with fla's own auto-registration (both call `register(..., exist_ok=True)`).
6. **DNA tokenizer must expose `bos`/`eos`** for varlen packing, and vocab-size propagation relies on either `tokenizer.vocab_size` (HF) or `get_vocab_size()` (custom) — mismatched embedding size vs config is a common footgun.
7. **Default validation paths are English-corpus specific** (wiki_val / C4); must override `--training.val_data_dir` for DNA or PPL eval will fail/mislead.

---

## 6. Key file reference

| Concern | Path |
|---|---|
| OpenToMe purpose | `OpenToMe/README.md:1`, `OpenToMe/setup.py:107` |
| Real dep stack (versions) | `OpenToMe/fla_environment.yml` |
| Vendored flame engine | `OpenToMe/trainer/flame/` |
| Training loop (patched) | `OpenToMe/trainer/flame/flame/train.py` |
| BACKBONE dispatch | `OpenToMe/trainer/flame/flame/train.py:43-70` |
| Custom tokenizer hook | `OpenToMe/trainer/flame/flame/train.py:390-401`; `OpenToMe/opentome/tokenizer/build_tokenizer.py` |
| Data pipeline | `OpenToMe/trainer/flame/flame/data.py` (`build_dataset:555`, `build_dataloader:736`, collator:321) |
| TOML defaults | `OpenToMe/trainer/flame/flame/models/fla.toml` |
| DeltaNet / Transformer configs | `OpenToMe/trainer/flame/configs/{delta_net_340M,transformer_340M}.json` |
| Byte-level recipes | `OpenToMe/trainer/flame/scripts/byte/{deltanet,transformer++}.sh` |
| Local LM model copies | `OpenToMe/opentome/models/{transformer,delta_net,...}/` |
| Launch script | `OpenToMe/trainer/flame/train.sh` |
| flame upstream (reference) | `flame/README.md`, `flame/custom_models/sba/` (custom-model template) |
| fla model registry | `flash-linear-attention/fla/models/__init__.py` |
| fla Transformer++ | `flash-linear-attention/fla/models/transformer/modeling_transformer.py:346` |
| fla DeltaNet | `flash-linear-attention/fla/models/delta_net/modeling_delta_net.py:342` |
| HNet (standalone) | `hnet/hnet/models/mixer_seq.py:22`, `hnet/hnet/models/config_hnet.py`, `hnet/pyproject.toml` |
