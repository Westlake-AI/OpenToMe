"""Shared runtime helpers for KV-compression inference evaluations."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from opentome.compress import CompressedDynamicCache, KVCompressionConfig, POLICY_REGISTRY
from opentome.models.kv_compression import patch_model_for_kv_compression


METHOD_CHOICES = ("none", *POLICY_REGISTRY)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--method", choices=METHOD_CHOICES, default="none")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="flash_attention_2")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)


def add_compression_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-capacity-prompt", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--pooling", choices=("avgpool", "maxpool"), default="maxpool")
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument("--pyramid-beta", type=float, default=0.5)
    parser.add_argument("--quest-page-size", type=int, default=16)
    parser.add_argument("--nacl-proxy-size", type=int, default=32)
    parser.add_argument(
        "--nacl-proxy-mode", choices=("suffix", "prefix", "edges"), default="suffix"
    )
    parser.add_argument("--nacl-random-budget", type=int, default=0)
    parser.add_argument("--scissorhands-decay", type=float, default=1.0)
    parser.add_argument(
        "--scissorhands-selection", choices=("topk", "prob"), default="topk"
    )
    parser.add_argument("--random-temperature", type=float, default=1.0)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device):
    if name == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return getattr(torch, name)


def load_model_and_tokenizer(args):
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()
    if args.method != "none":
        patch_model_for_kv_compression(model)
    return model, tokenizer, device


def compression_config(args, model) -> KVCompressionConfig:
    return KVCompressionConfig(
        method=args.method,
        max_capacity_prompt=args.max_capacity_prompt,
        window_size=args.window_size,
        kernel_size=args.kernel_size,
        pooling=args.pooling,
        sink_size=args.sink_size,
        pyramid_beta=args.pyramid_beta,
        num_hidden_layers=model.config.num_hidden_layers,
        quest_page_size=args.quest_page_size,
        nacl_proxy_size=args.nacl_proxy_size,
        nacl_proxy_mode=args.nacl_proxy_mode,
        nacl_random_budget=args.nacl_random_budget,
        scissorhands_decay=args.scissorhands_decay,
        scissorhands_selection=args.scissorhands_selection,
        random_seed=args.seed,
        random_temperature=args.random_temperature,
    )


def build_cache(args, model):
    if args.method == "none":
        return DynamicCache()
    return CompressedDynamicCache(compression_config(args, model))


def cache_stats(cache):
    if isinstance(cache, CompressedDynamicCache):
        return {
            "cache_layer_lengths": cache.layer_lengths(),
            "cache_bytes": cache.cache_bytes(),
            "logical_cache_length": cache.get_logical_length(),
        }
    lengths = [pair[0].shape[-2] for pair in cache]
    size = sum(t.numel() * t.element_size() for pair in cache for t in pair)
    logical_length = cache.get_seq_length() if hasattr(cache, "get_seq_length") else (lengths[0] if lengths else 0)
    return {"cache_layer_lengths": lengths, "cache_bytes": size, "logical_cache_length": logical_length}


def middle_truncate(input_ids: torch.Tensor, max_length: int) -> torch.Tensor:
    if input_ids.shape[-1] <= max_length:
        return input_ids
    left = max_length // 2
    return torch.cat((input_ids[..., :left], input_ids[..., -(max_length - left):]), dim=-1)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_metadata(args):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "args": vars(args),
    }


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
