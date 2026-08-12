import argparse
import json
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from opentome.compress import CompressedDynamicCache, KVCompressionConfig, POLICY_REGISTRY
from opentome.models.kv_compression import patch_model_for_kv_compression

from evaluations.inference.kv_utils import run_metadata, write_json
from evaluations.inference.longbench.data import load_longbench_records, normalize_dataset_args


CONFIG_DIR = Path(__file__).with_name("config")
NO_CHAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LongBench predictions with KV compression")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", action="append", help="Dataset, comma list, or 'all'")
    parser.add_argument("--longbench-e", action="store_true")
    parser.add_argument("--dataset-path", default=str(REPO_ROOT / "data" / "LongBench"))
    parser.add_argument("--local-data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "work_dirs" / "longbench")
    parser.add_argument("--run-name")
    parser.add_argument("--method", choices=("none", *POLICY_REGISTRY), default="none")
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chat-template", choices=("auto", "none", "llama2"), default="auto")
    parser.add_argument("--max-capacity-prompt", type=int, default=2048)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--pooling", choices=("avgpool", "maxpool"), default="avgpool")
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument("--pyramid-beta", type=float, default=0.5)
    parser.add_argument("--quest-page-size", type=int, default=16)
    parser.add_argument("--nacl-proxy-size", type=int, default=32)
    parser.add_argument("--nacl-proxy-mode", choices=("suffix", "prefix", "edges"), default="suffix")
    parser.add_argument("--nacl-random-budget", type=int, default=0)
    parser.add_argument("--scissorhands-decay", type=float, default=1.0)
    parser.add_argument("--scissorhands-selection", choices=("topk", "prob"), default="topk")
    parser.add_argument("--random-temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="flash_attention_2")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_configs():
    prompts = json.loads((CONFIG_DIR / "dataset2prompt.json").read_text(encoding="utf-8"))
    max_lengths = json.loads((CONFIG_DIR / "dataset2maxlen.json").read_text(encoding="utf-8"))
    return prompts, max_lengths


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def resolve_dtype(name, device):
    if name == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return getattr(torch, name)


def build_chat_prompt(tokenizer, prompt, dataset, mode):
    if mode == "none" or dataset in NO_CHAT_DATASETS:
        return prompt
    if mode == "llama2":
        return f"[INST] {prompt} [/INST]"
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    return prompt


def middle_truncate(input_ids, max_length):
    if input_ids.shape[-1] <= max_length:
        return input_ids
    left = max_length // 2
    right = max_length - left
    return torch.cat((input_ids[..., :left], input_ids[..., -right:]), dim=-1)


def format_prompt(record, template):
    values = dict(record)
    values.setdefault("input", values.get("question", ""))
    values.setdefault("context", "")
    return template.format(**values)


def cache_bytes(cache):
    if isinstance(cache, CompressedDynamicCache):
        return cache.cache_bytes()
    return sum(tensor.numel() * tensor.element_size() for pair in cache for tensor in pair)


def build_cache(args, model):
    if args.method == "none":
        return DynamicCache()
    return CompressedDynamicCache(
        KVCompressionConfig(
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
    )


def prediction_path(args, dataset):
    run_name = args.run_name or Path(args.model_path).name
    variant = "longbench_e" if args.longbench_e else "longbench"
    return (
        args.output_dir
        / run_name
        / args.method
        / str(args.max_capacity_prompt)
        / variant
        / f"{dataset}.jsonl"
    )


def existing_record_count(path):
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


@torch.inference_mode()
def predict_dataset(model, tokenizer, records, dataset, prompt_template, max_new_tokens, args, device):
    output_path = prediction_path(args, dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    skip = existing_record_count(output_path)
    model_limit = args.max_context_length or int(getattr(model.config, "max_position_embeddings", 4096))
    prompt_limit = max(1, model_limit - max_new_tokens)
    total = len(records) if hasattr(records, "__len__") else None
    if args.max_samples is not None:
        total = min(total, args.max_samples) if total is not None else args.max_samples

    written = 0
    with output_path.open("a", encoding="utf-8") as output_handle:
        for index, record in enumerate(tqdm(records, total=total, desc=dataset)):
            if args.max_samples is not None and index >= args.max_samples:
                break
            if index < skip:
                continue
            prompt = format_prompt(record, prompt_template)
            prompt = build_chat_prompt(tokenizer, prompt, dataset, args.chat_template)
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
            input_ids = middle_truncate(input_ids, prompt_limit).to(device)
            attention_mask = torch.ones_like(input_ids)
            cache = build_cache(args, model)

            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": cache,
                "use_cache": True,
                "max_new_tokens": max_new_tokens,
                "num_beams": 1,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            }
            if dataset == "samsum":
                newline_ids = tokenizer.encode("\n", add_special_tokens=False)
                eos_ids = [tokenizer.eos_token_id]
                if newline_ids:
                    eos_ids.append(newline_ids[-1])
                generation_kwargs["eos_token_id"] = [token for token in eos_ids if token is not None]

            start = time.perf_counter()
            generated = model.generate(**generation_kwargs)[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            prediction = tokenizer.decode(generated[input_ids.shape[-1]:], skip_special_tokens=True)
            layer_lengths = (
                cache.layer_lengths() if isinstance(cache, CompressedDynamicCache)
                else [pair[0].shape[-2] for pair in cache]
            )
            result = {
                "pred": prediction,
                "answers": record.get("answers", []),
                "all_classes": record.get("all_classes", []),
                "length": record.get("length", input_ids.shape[-1]),
                "input_tokens": input_ids.shape[-1],
                "generated_tokens": generated.shape[-1] - input_ids.shape[-1],
                "generation_seconds": elapsed,
                "cache_layer_lengths": layer_lengths,
                "cache_bytes": cache_bytes(cache),
                "method": args.method,
            }
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_handle.flush()
            written += 1
    return output_path, written, skip


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    metadata = run_metadata(args)
    datasets = normalize_dataset_args(args.dataset, args.longbench_e)
    if args.local_data and args.local_data.is_file() and len(datasets) != 1:
        raise ValueError("A --local-data file can only be used with one --dataset")
    prompts, generation_lengths = load_configs()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
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

    variant = "longbench_e" if args.longbench_e else "longbench"
    run_name = args.run_name or Path(args.model_path).name
    write_json(
        args.output_dir
        / run_name
        / args.method
        / str(args.max_capacity_prompt)
        / variant
        / "metadata.json",
        metadata,
    )

    summaries = []
    for dataset in datasets:
        records = load_longbench_records(
            dataset, args.longbench_e, args.local_data, args.dataset_path
        )
        path, written, resumed = predict_dataset(
            model, tokenizer, records, dataset, prompts[dataset],
            generation_lengths[dataset], args, device,
        )
        summaries.append({"dataset": dataset, "path": str(path), "written": written, "resumed": resumed})
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
