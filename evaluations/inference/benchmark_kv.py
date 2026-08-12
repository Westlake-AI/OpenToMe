"""Latency, throughput, memory, and cache-size benchmark for KV policies."""

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import torch

from kv_utils import (
    add_compression_args,
    add_model_args,
    build_cache,
    cache_stats,
    load_model_and_tokenizer,
    run_metadata,
    synchronize,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    add_compression_args(parser)
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", default="Summarize KV cache compression in one paragraph.\n")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompt-repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Result JSON path (default: work_dirs/benchmark_kv/"
            "<model>/<method>/<max-capacity-prompt>/result.json)"
        ),
    )
    return parser.parse_args()


def default_output_path(args):
    return (
        REPO_ROOT
        / "work_dirs"
        / "benchmark_kv"
        / Path(args.model_path).name
        / args.method
        / str(args.max_capacity_prompt)
        / "result.json"
    )


@torch.inference_mode()
def run_once(model, tokenizer, prompt, args, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    cache = build_cache(args, model)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    synchronize(device)
    elapsed = time.perf_counter() - start
    generated = output.shape[-1] - input_ids.shape[-1]
    result = {
        "input_tokens": input_ids.shape[-1],
        "generated_tokens": generated,
        "elapsed_seconds": elapsed,
        "generated_tokens_per_second": generated / elapsed if elapsed else None,
        **cache_stats(cache),
    }
    if device.type == "cuda":
        result.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    return result


def main():
    args = parse_args()
    if args.output is None:
        args.output = default_output_path(args)
    if args.prompt_repeats <= 0 or args.repeat <= 0 or args.warmup < 0:
        raise ValueError("prompt-repeats and repeat must be positive; warmup must be non-negative")
    torch.manual_seed(args.seed)
    metadata = run_metadata(args)
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    prompt *= args.prompt_repeats
    model, tokenizer, device = load_model_and_tokenizer(args)
    for _ in range(args.warmup):
        run_once(model, tokenizer, prompt, args, device)
    runs = [run_once(model, tokenizer, prompt, args, device) for _ in range(args.repeat)]
    rates = [run["generated_tokens_per_second"] for run in runs]
    result = {
        "metadata": metadata,
        "runs": runs,
        "mean_generated_tokens_per_second": sum(rates) / len(rates),
    }
    if device.type == "cuda":
        result["max_peak_allocated_bytes"] = max(run["peak_allocated_bytes"] for run in runs)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
