# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import time

# Per-rank Triton/Inductor cache. Shared /tmp/triton_cache_* under DDP + torch.compile
# races in torch._inductor (FileExistsError on os.replace); set before import torch.
# _lr_triton = os.environ.get("LOCAL_RANK")
# if _lr_triton is not None:
#     _td = os.environ.get("TRITON_CACHE_DIR", "/tmp/triton_inductor")
#     os.environ["TRITON_CACHE_DIR"] = f"{_td}_r{_lr_triton}"

from datetime import timedelta

import fla  # noqa
import torch
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.ops.utils import prepare_position_ids
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.ft import FTParallelDims, init_ft_manager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.metrics import build_device_memory_monitor, build_metrics_processor, ensure_pp_loss_visible
# from torchtitan.components.optimizer import build_optimizers
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import TrainSpec, get_train_spec, register_train_spec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ------ jinxin added ------ #
backbone = os.environ.get("BACKBONE", "None")
print("*" * 50)
if "gated_deltanet" in backbone:
    print("Gated-DeltaNet")
    import opentome.models.gated_deltanet
elif "delta_net" in backbone:
    print("DeltaNet")
    import opentome.models.delta_net
elif "gla" in backbone:
    print("GLA")
    import opentome.models.gla
elif "transformer++" in backbone:
    print("Transformer++")
    import opentome.models.transformer
elif "gsa" in backbone:
    print("GSA")
    import opentome.models.gsa
elif "qwen3_next" in backbone:
    print("Qwen3-NeXt")
    import opentome.models.qwen3_next
elif "blt" in backbone:
    print("BLT")
    import opentome.models.blt
elif "mergenet" in backbone:
    print("MergeNet")
    import opentome.models.mergenet_nlp
else:
    print("None")
# --- Tokenizer & Optimizer --- #
tokenizer_name = os.environ.get("TOKENIZER_NAME", "default")
default_opt = os.environ.get("DEFAULT_OPT", "AdamW")
if default_opt in ["AdamW", "Adam"]:    # these are the default optimizers in flame
    from torchtitan.components.optimizer import build_optimizers
else:
    from opentome.utils.optimization import build_optimizers
print(f"Tokenizer name: {tokenizer_name}")
print(f"Optimizer: {default_opt}")
from opentome.tokenizer.build_tokenizer import TokenizerArgs
print("*" * 50)
# ------ End of jinxin added ------ #

from flame.components.checkpoint import TrainState
from flame.config_manager import JobConfig
from flame.data import build_dataloader, build_dataset
from flame.models.parallelize_fla import parallelize_fla
from flame.models.pipeline_fla import pipeline_fla
from flame.tools.utils import get_nparams_and_flops

def build_tokenizer(job_config: JobConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(job_config.model.tokenizer_path)


# ------ jinxin ------ #
def build_val_chunks_cache(
    val_data_path: str,
    tokenizer,
    seq_len: int,
    world_mesh,
    parallel_dims,
    c4_target_eval_tokens: int = 10_000_000,
) -> "list[list[int]]":
    """
    Load, tokenize, and chunk the validation set exactly once before training.
    Returns a list of token-id chunks (each of length seq_len) for this dp rank.

    - parquet (wiki_val): full set, sharded by dp_rank
    - json.gz glob (C4): streaming, capped at c4_target_eval_tokens per rank
    """
    if parallel_dims.dp_enabled:
        dp_degree = world_mesh["dp"].size()
        dp_rank   = world_mesh["dp"].get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    is_c4 = not val_data_path.endswith(".parquet")

    # 1. Load texts
    if not is_c4:
        import pandas as pd
        df = pd.read_parquet(val_data_path)
        texts = df["text"].tolist()[dp_rank::dp_degree]
    else:
        from datasets import load_dataset
        ds = load_dataset("json", data_files=val_data_path, split="train", streaming=True)
        texts = []
        tok_est = 0
        for idx, sample in enumerate(ds):
            if idx % dp_degree != dp_rank:
                continue
            texts.append(sample["text"])
            tok_est += len(sample["text"]) // 4  # ~4 chars per token
            if c4_target_eval_tokens > 0 and tok_est >= c4_target_eval_tokens + seq_len:
                break

    # 2. Batched tokenization (512 texts per call)
    token_buffer: list[int] = []
    for i in range(0, len(texts), 512):
        batch_ids = tokenizer(
            texts[i : i + 512],
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]
        for ids in batch_ids:
            token_buffer.extend(ids)
        if is_c4 and c4_target_eval_tokens > 0 and len(token_buffer) >= c4_target_eval_tokens + seq_len:
            break

    # 3. Cut into non-overlapping seq_len chunks
    chunks = [
        token_buffer[i : i + seq_len]
        for i in range(0, len(token_buffer) - seq_len, seq_len)
    ]
    return chunks


def evaluate_ppl(
    model,
    tokenizer,
    val_data_path: str,
    batch_size: int,
    seq_len: int,
    device,
    device_type: str,  # kept for call-site compatibility, unused internally
    world_mesh,
    parallel_dims,
    maybe_enable_amp,
    color,
    step: int,
    chunks_cache: "list[list[int]] | None" = None,
) -> float:
    """
    Evaluation function for PPL on validation set.

    When chunks_cache is provided (pre-built in main before the train loop),
    all IO and tokenization is skipped — only the forward pass runs.
    This is the fast path used during training.

    Without cache (chunks_cache=None), falls back to loading from disk:
      - parquet (wiki_val): full set, no token budget
      - json.gz glob (C4): streaming, 10M-token budget per rank
    """
    model.eval()

    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    # ------------------------------------------------------------------ #
    # 1. Get chunks — from cache (fast) or disk (slow, fallback only)     #
    # ------------------------------------------------------------------ #
    if chunks_cache is not None:
        chunks = chunks_cache  # zero IO, zero tokenization
    else:
        # Fallback: load from disk (only used if cache was not built)
        C4_TARGET_EVAL_TOKENS = 10_000_000
        is_c4 = not val_data_path.endswith(".parquet")

        if not is_c4:
            import pandas as pd
            df = pd.read_parquet(val_data_path)
            texts_for_rank = df["text"].tolist()[dp_rank::dp_degree]
        else:
            from datasets import load_dataset
            ds = load_dataset("json", data_files=val_data_path, split="train", streaming=True)
            texts_for_rank = []
            token_estimate = 0
            for idx, sample in enumerate(ds):
                if idx % dp_degree != dp_rank:
                    continue
                texts_for_rank.append(sample["text"])
                token_estimate += len(sample["text"]) // 4
                if C4_TARGET_EVAL_TOKENS > 0 and token_estimate >= C4_TARGET_EVAL_TOKENS + seq_len:
                    break

        TOKENIZE_BATCH = 512
        token_buffer = []
        for i in range(0, len(texts_for_rank), TOKENIZE_BATCH):
            batch_ids = tokenizer(
                texts_for_rank[i : i + TOKENIZE_BATCH],
                return_attention_mask=False,
                add_special_tokens=False,
            )["input_ids"]
            for ids in batch_ids:
                token_buffer.extend(ids)
            if is_c4 and C4_TARGET_EVAL_TOKENS > 0 and len(token_buffer) >= C4_TARGET_EVAL_TOKENS + seq_len:
                break

        chunks = [
            token_buffer[i : i + seq_len]
            for i in range(0, len(token_buffer) - seq_len, seq_len)
        ]

    # ------------------------------------------------------------------ #
    # 2. Forward pass over all chunks                                     #
    # ------------------------------------------------------------------ #
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0, device=device, dtype=torch.long)

    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            if not batch_chunks:
                continue
            input_ids = torch.tensor(batch_chunks, dtype=torch.long, device=device)
            labels = input_ids.clone()
            position_ids = (
                torch.arange(0, input_ids.shape[1], device=device)
                .repeat(input_ids.shape[0], 1)
                .to(torch.int32)
            )
            with maybe_enable_amp:
                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    position_ids=position_ids,
                )
            n_tokens = labels.numel()
            total_loss += output.loss.detach() * n_tokens
            total_tokens += n_tokens

    # ------------------------------------------------------------------ #
    # 3. All-reduce across dp ranks                                       #
    # ------------------------------------------------------------------ #
    if parallel_dims.dp_enabled:
        torch.distributed.all_reduce(total_loss, group=world_mesh["dp"].get_group())
        torch.distributed.all_reduce(total_tokens, group=world_mesh["dp"].get_group())

    avg_loss = (
        (total_loss / total_tokens).item()
        if total_tokens.item() > 0
        else float("inf")
    )
    ppl = math.exp(avg_loss)

    model.train()
    return ppl


register_train_spec(
    TrainSpec(
        name="fla",
        cls=AutoModelForCausalLM,
        config=AutoConfig,
        parallelize_fn=parallelize_fla,
        pipelining_fn=pipeline_fla,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.experimental.custom_model_path:
        utils.import_module_from_path(job_config.experimental.custom_model_path)

    # used for colorful printing
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    if job_config.job.print_args:
        logger.info(
            f"{color.green}{json.dumps(job_config.to_dict(), indent=2, sort_keys=True)}{color.reset}"
        )

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    device_module, device_type = utils.device_module, utils.device_type
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(device)
    ft_manager = init_ft_manager(job_config)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    if not ft_manager.enabled:
        parallel_dims = ParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
        )
    else:
        parallel_dims = FTParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
            ft_manager=ft_manager,
        )
    dist_utils.init_distributed(job_config)
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Pipeline parallelism is not supported in this version"
        )
        """
        ! TODO[flame]: We need to fix the pipeline parallelism for flame
        [x] Match the key of models' components with the actual naming
        [ ] Fix the post-init and tie-embedding for pipeline parallelism, HF's transformer automatically
            forces to tie if head is None, we need to handle this case
        [ ]
        """
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    dist_utils.set_determinism(
        world_mesh, device, job_config.training.seed, job_config.training.deterministic
    )
    train_spec = get_train_spec(job_config.model.name)

    logger.info("Loading tokenizer...")
    # ------ jinxin ------ #
    if tokenizer_name =="default":
        tokenizer = AutoTokenizer.from_pretrained(
            job_config.model.tokenizer_path,
            trust_remote_code=True,
            model_max_length=int(1e10),
        )
        logger.info(f"{tokenizer}")
    else:
        assert tokenizer_name in ["bytes", "sentencepiece", "tiktoken", "blt"], f"Invalid tokenizer name: {tokenizer_name}"
        tokenizer = TokenizerArgs(
            name=tokenizer_name,
            init_kwargs={
                "vocab_size_unit_1": 256,
                "bpe_delim": False,
                "bpe_tokenizer_path": job_config.model.tokenizer_path,
                "add_bos": True,
                "add_eos": True,
            }
        ).build()

    logger.info(f"Loading model config from {job_config.model.config}")
    model_config = AutoConfig.from_pretrained(job_config.model.config)
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. disable fused norm if TP is enabled
    # 3. vocab size from tokenizer
    # 4. context_len base on inputs
    if parallel_dims.tp_enabled:
        if model_config.fuse_norm:
            logger.warning(
                f"{color.red}"
                f"Fused norm is not compatible with tensor parallelism. "
                f"Disabling it for now."
                f"{color.reset}"
            )
            model_config.fuse_norm = False
    if parallel_dims.loss_parallel_enabled:
        if model_config.fuse_linear_cross_entropy:
            logger.warning(
                f"{color.red}"
                f"Loss parallel enabled. Disabling fused cross entropy for now."
                f"{color.reset}"
            )
            model_config.fuse_linear_cross_entropy = False

    # ------ jinxin ------ #
    if tokenizer_name =="default":
        model_config.vocab_size = max(tokenizer.vocab_size, model_config.vocab_size)
    else:
        assert tokenizer_name in ["bytes", "sentencepiece", "tiktoken", "blt"], f"Invalid tokenizer name: {tokenizer_name}"
        model_config.vocab_size = tokenizer.get_vocab_size()
    logger.info(
        f"Building model from the config\n{color.green}{model_config}{color.reset}"
    )
    # ------ Modeling ------ #
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)
        if (
            getattr(model_config, "fuse_linear_cross_entropy", False)
            and FusedLinearCrossEntropyLoss is not None
        ):
            model.criterion = FusedLinearCrossEntropyLoss(
                num_chunks=8 // parallel_dims.tp
            )
        # defer weight initialization until after parallelisms are applied
        model.apply(lambda m: setattr(m, "_is_hf_initialized", False))
    logger.info(f"{color.blue}\n{model}{color.reset}\n")

    # Build the collection of model converters. No-op if `model.converters` empty
    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)

    # calculate model size and flops per token
    model_param_count, num_flops_per_token = get_nparams_and_flops(
        model, model_config, job_config.training.context_len
    )

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
    else:
        init_device = device_type

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
        (
            pp_schedule,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = train_spec.pipelining_fn(
            model,
            pp_mesh,
            parallel_dims,
            job_config,
            device,
            model_config,
            train_spec.loss_fn,
        )
        # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
        del model

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            train_spec.parallelize_fn(m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            with torch.no_grad():
                m.post_init()
            m.train()

        # confirm that user will be able to view loss metrics on the console
        ensure_pp_loss_visible(parallel_dims, job_config, color)
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.post_init()
        model.train()

        model_parts = [model]

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    # SCALE / RMNP need precomputed parameter groups; Muon uses local opentome/optimizer/muon.py
    # (param split inside Muon.__init__, not build_muon_metrics).
    metric_model = model_parts[0]
    if default_opt == "SCALE":
        from opentome.utils.optimization import build_scale_metrics

        metrics = build_scale_metrics(metric_model)
        optimizers = train_spec.build_optimizers_fn(
            model_parts, job_config, ft_manager, metrics=metrics
        )
    elif default_opt == "RMNP":
        from opentome.utils.optimization import build_rmnp_metrics

        metrics = build_rmnp_metrics(metric_model)
        optimizers = train_spec.build_optimizers_fn(
            model_parts, job_config, ft_manager, metrics=metrics
        )
    else:
        optimizers = train_spec.build_optimizers_fn(model_parts, job_config, ft_manager)
    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
    # Post optimizer step model converters hook.
    # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
    # where it issues a single all-reduce for all parameters at once for better performance
    optimizers.register_step_post_hook(
        lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
    )

    train_state = TrainState()

    logger.info(
        f"Loading dataset {job_config.training.dataset}"
        f":{job_config.training.dataset_name}"
        if job_config.training.dataset_name is not None
        else ""
    )
    dataset = build_dataset(
        dataset=job_config.training.dataset,
        dataset_name=job_config.training.dataset_name,
        dataset_split=job_config.training.dataset_split,
        data_dir=job_config.training.data_dir,
        data_files=job_config.training.data_files,
        data_probs=job_config.training.data_probs,
        streaming=job_config.training.streaming,
        dp_degree=dp_degree,
        num_workers=job_config.training.num_workers,
        seed=job_config.training.seed,
    )

    logger.info("Building dataloader...")
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=dp_rank,
        world_size=dp_degree,
        batch_size=job_config.training.batch_size,
        seq_len=job_config.training.seq_len,
        context_len=job_config.training.context_len,
        varlen=job_config.training.varlen,
        num_workers=job_config.training.num_workers,
        pin_memory=job_config.training.pin_memory,
        persistent_workers=job_config.training.persistent_workers,
        snapshot_every_n_steps=job_config.checkpoint.interval,
    )

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=dataloader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
        ft_manager=ft_manager,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert world_size == 1, (
            "Must create seed checkpoint using a single device, to disable sharding"
        )
        assert job_config.checkpoint.enable_checkpoint, (
            "Must enable checkpointing when creating a seed checkpoint"
        )
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=job_config.checkpoint.load_step)
    metric_logger = build_metrics_processor(job_config, parallel_dims)
    # Set dependent attributes for metric_logger
    metric_logger.num_flops_per_token = num_flops_per_token
    metric_logger.optimizers = optimizers  # Pass optimizers if needed by logger logic
    metric_logger.lr_schedulers = (
        lr_schedulers  # Pass schedulers if needed by logger logic
    )

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0 and len(metric_logger.data_loading_times) > 0:
        for idx, step in enumerate(train_state.log_steps):
            metric_logger.log(
                step,
                global_avg_loss=train_state.global_avg_losses[idx],
                global_max_loss=train_state.global_max_losses[idx],
            )

    data_iterator = iter(dataloader)

    train_context = dist_utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )
    maybe_enable_amp = dist_utils.maybe_enable_amp(
        parallel_dims,
        job_config.training.mixed_precision_param,
        device_type,
    )

    # variables used to keep info for metrics logging
    device_memory_monitor.reset_peak_stats()

    global_batch_size = (
        job_config.training.batch_size
        * dp_degree
        * job_config.training.gradient_accumulation_steps
    )
    num_tokens_per_step = global_batch_size * job_config.training.seq_len

    # ------ jinxin ------ #
    # PPL related configurations for validation
    val_times = getattr(job_config.training, "val_times", 0)
    val_interval = (job_config.training.steps // val_times) if val_times > 0 else 0
    val_data_dir = getattr(job_config.training, "val_data_dir", None)

    if val_data_dir is not None:    # User explicitly specified val data path
        if os.path.isfile(val_data_dir):    # Directly passed file path (e.g. parquet)
            val_parquet = val_data_dir
        elif val_data_dir.endswith(".json.gz") or "*" in val_data_dir:    # Passed is a glob pattern
            val_parquet = val_data_dir
        else:    # Passed is a directory, automatically find parquet file
            val_parquet = os.path.join(val_data_dir, "validation-00000-of-00001.parquet")
    else:
        # Not specified val_data_dir, automatically select based on training dataset
        training_dataset = getattr(job_config.training, "dataset", "")
        training_data_files = getattr(job_config.training, "data_files", "") or ""
        training_data_dir_cfg = getattr(job_config.training, "data_dir", "") or ""
        is_c4 = (
            training_dataset == "json" and ("c4" in training_data_files.lower() or "c4" in training_data_dir_cfg.lower())
        )
        if is_c4:    # C4 training, default to use C4 built-in validation set
            if training_data_files and "c4" in training_data_files.lower():
                c4_dir = os.path.dirname(training_data_files.split(",")[0])
            else:
                c4_dir = training_data_dir_cfg.split(",")[0]
            val_parquet = os.path.join(c4_dir, "c4-validation.*.json.gz")
            logger.info(f"{color.yellow}C4 training detected: using C4 built-in validation set for PPL eval: {val_parquet}{color.reset}")
        else:   # Default to use wiki_val
            val_parquet = os.path.join(os.getcwd(), "data/wiki_val", "validation-00000-of-00001.parquet")

    # ------ jinxin ------ #
    # Pre-cache val chunks once before training — each evaluate_ppl call reuses
    # this cache with zero IO / tokenization overhead.
    C4_TARGET_EVAL_TOKENS = 10_000_000
    logger.info(f"{color.yellow}Pre-caching val chunks from {val_parquet} ...{color.reset}")
    val_chunks_cache = build_val_chunks_cache(
        val_data_path=val_parquet,
        tokenizer=tokenizer,
        seq_len=job_config.training.seq_len,
        world_mesh=world_mesh,
        parallel_dims=parallel_dims,
        c4_target_eval_tokens=C4_TARGET_EVAL_TOKENS,
    )
    logger.info(
        f"{color.yellow}Val cache ready: {len(val_chunks_cache)} chunks "
        f"({len(val_chunks_cache) * job_config.training.seq_len:,} tokens per rank){color.reset}"
    )

    # train loop
    logger.info(f"{color.red}***** Running training *****{color.reset}")
    logger.info(f"{color.green}  Training starts at step {train_state.step + 1}")
    logger.info(
        f"{color.green}  Number of tokens per sequence = {job_config.training.seq_len:,}"
    )
    logger.info(
        f"{color.green}  Gradient Accumulation steps = {job_config.training.gradient_accumulation_steps}"
    )
    logger.info(
        f"{color.green}  Instantaneous batch size (per device) = {job_config.training.batch_size:,}"
    )
    logger.info(
        f"{color.green}  Global batch size (w. parallel, distributed & accumulation) = {global_batch_size:,}"
        f" ({num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Total optimization steps = {job_config.training.steps:,} "
        f"({job_config.training.steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Warmup steps = {job_config.lr_scheduler.warmup_steps:,}"
        f" ({job_config.lr_scheduler.warmup_steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Number of parameters = {model_param_count:,} {color.reset}"
    )

    # ------ jinxin ------ #
    if val_interval > 0:
        logger.info(
            f"{color.green}Val PPL will be computed every {val_interval} steps "
            f"({val_times} times total). Val data: {val_parquet}{color.reset}"
        )

    with (
        maybe_enable_profiling(
            job_config, global_step=train_state.step
        ) as torch_profiler,
        maybe_enable_memory_snapshot(
            job_config, global_step=train_state.step
        ) as memory_profiler,
    ):
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            optimizers.zero_grad()

            losses = []
            # do gradient accumulation if enabled
            for _ in range(job_config.training.gradient_accumulation_steps):
                # get batch
                data_load_start = time.perf_counter()
                batch = next(data_iterator)
                input_ids, labels = batch["input_ids"], batch["labels"]

                # Update metrics processor state before forward/backward
                metric_logger.ntokens_since_last_log += labels.numel()
                metric_logger.data_loading_times.append(
                    time.perf_counter() - data_load_start
                )

                input_ids = input_ids.to(device_type)

                """
                TODO[flame]: We need to carefully handle the position_ids for TP/CP
                Depending on the Models'PE, the position_ids might be different.

                e.g. for TP
                    For RoPE, all ranks have the same position_ids. [FOR HF model]
                    For sinusoidal, each rank has the coresponding chunked  position_ids. [FOR HF model]

                e.g. for CP, [optional_context_parallel_ctx shoudl automatically distbute the position_ids]
                    Each rank has the coresponding chunked position_ids. [FOR All model]

                """
                labels = labels.to(device_type)
                cu_seqlens = (
                    batch["cu_seqlens"].to(device_type)
                    if "cu_seqlens" in batch
                    else None
                )
                if cu_seqlens is not None:
                    position_ids = prepare_position_ids(cu_seqlens).to(torch.int32)
                else:
                    position_ids = (
                        torch.arange(0, input_ids.shape[1], device=device_type)
                        .repeat(input_ids.shape[0], 1)
                        .to(torch.int32)
                    )
                # apply context parallelism if cp is enabled
                # ensure CP handles the separate freqs_cis buffer for each pp stage
                optional_context_parallel_ctx = (
                    dist_utils.create_context_parallel_ctx(
                        cp_mesh=world_mesh["cp"],
                        cp_buffers=[input_ids, labels, position_ids],
                        cp_seq_dims=[1, 1, 1],
                        cp_no_restore_buffers={input_ids, labels, position_ids},
                        cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                    )
                    if parallel_dims.cp_enabled
                    else None
                )

                # #! TODO[flame], we should distribute the position_ids as well with CP
                if parallel_dims.pp_enabled:
                    raise NotImplementedError(
                        "Pipeline parallelism is not supported in this version"
                    )
                    # Pipeline Parallel forward / backward inside step() call
                    with train_context(optional_context_parallel_ctx):
                        targets, losses = (
                            (labels, []) if has_last_stage else (None, None)
                        )

                        if has_first_stage:
                            pp_schedule.step(input_ids, target=targets, losses=losses)
                        else:
                            pp_schedule.step(target=targets, losses=losses)

                    # accumulate losses across pipeline microbatches
                    # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                    loss = (
                        torch.mean(torch.stack(losses)).to(device)
                        if has_last_stage
                        else torch.tensor([-1.0], device=device)
                    )
                else:
                    # Non-PP forward / backward
                    with train_context(optional_context_parallel_ctx):
                        with maybe_enable_amp:   # --- jinxin --- #
                            output = model(
                                input_ids=input_ids,
                                labels=labels,
                                position_ids=position_ids,
                                cu_seqlens=cu_seqlens,
                        )
                        loss = (
                            output.loss
                            / job_config.training.gradient_accumulation_steps
                        )
                        loss.backward()

                losses.append(loss)
            loss = sum(losses)

            # clip gradients
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            if job_config.training.skip_nan_inf and (
                grad_norm.isnan() or grad_norm.isinf()
            ):
                logger.warning(
                    f"Skipping optimizer step - detected invalid gradient norm: {grad_norm:.4f}"
                )
                optimizers.zero_grad()
                train_state.skipped_step += 1
            else:
                optimizers.step()
            lr_schedulers.step()

            # log metrics - Use MetricsProcessor
            if metric_logger.should_log(train_state.step):
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    # Use dist_mean/max on the accumulated loss for the step
                    global_avg_loss, global_max_loss = (
                        dist_utils.dist_mean(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                        dist_utils.dist_max(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                    )
                else:
                    # Scale back the loss before logging
                    global_avg_loss = global_max_loss = loss.item()

                # Update train state tokens and elapsed time
                time_now = time.perf_counter()
                time_delta = (
                    time_now - metric_logger.time_last_log
                )  # Use metric_logger's time
                train_state.token += (
                    metric_logger.ntokens_since_last_log  # Use tokens tracked by metric_logger
                    * parallel_dims.world_size
                    / parallel_dims.non_data_parallel_size
                )
                train_state.elapsed += timedelta(seconds=time_delta)
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                # Log using the metric processor
                last_lr = lr_schedulers.schedulers[0].get_last_lr()[0]
                eta = (
                    train_state.elapsed
                    * (job_config.training.steps - train_state.step)
                    / train_state.step
                )
                metric_logger.log(
                    train_state.step,
                    global_avg_loss,
                    global_max_loss,
                    extra_metrics={
                        "optimizer/lr": last_lr,
                        "optimizer/grad_norm": grad_norm.item(),
                        "optimizer/skipped_step": train_state.skipped_step,
                    },
                )

                logger.info(
                    f"{color.blue}lr: {last_lr:.4e} gnorm: {grad_norm:5.2f} "
                    f"{color.magenta}[{str(train_state.elapsed).split('.')[0]:>8}<{str(eta).split('.')[0]:>8}]{color.reset}"
                )

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            # ------ jinxin ------ #
            # Compute validation PPL during training by val_interval
            # if val_interval > 0 and train_state.step % val_interval == 0:
            if val_interval - 1 > 0 and train_state.step % val_interval == 0:
                val_ppl = evaluate_ppl(
                    model=model_parts[0],
                    tokenizer=tokenizer,
                    val_data_path=val_parquet,
                    batch_size=job_config.training.batch_size,
                    seq_len=job_config.training.seq_len,
                    device=device,
                    device_type=device_type,
                    world_mesh=world_mesh,
                    parallel_dims=parallel_dims,
                    maybe_enable_amp=maybe_enable_amp,
                    color=color,
                    step=train_state.step,
                    chunks_cache=val_chunks_cache,
                )
                if torch.distributed.get_rank() == 0:
                    logger.info(
                        f"{color.cyan}[Val PPL] step={train_state.step} | PPL={val_ppl:.4f}{color.reset}"
                    )
                metric_logger.logger.log({"val/ppl": val_ppl}, train_state.step)

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                dist_utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    # ------ jinxin ------ #
    # Compute final validation PPL on wiki_val after training
    logger.info("Computing final validation PPL on wiki_val...")
    final_val_ppl = evaluate_ppl(
        model=model_parts[0],
        tokenizer=tokenizer,
        val_data_path=val_parquet,
        batch_size=job_config.training.batch_size,
        seq_len=job_config.training.seq_len,
        device=device,
        device_type=device_type,
        world_mesh=world_mesh,
        parallel_dims=parallel_dims,
        maybe_enable_amp=maybe_enable_amp,
        color=color,
        step=job_config.training.steps,
        chunks_cache=val_chunks_cache,
    )
    if torch.distributed.get_rank() == 0:
        logger.info(
            f"{color.cyan}[Final Val PPL] step={job_config.training.steps} | PPL={final_val_ppl:.4f}{color.reset}"
        )
    metric_logger.logger.log({"val/ppl_final": final_val_ppl}, job_config.training.steps)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    init_logger()
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
