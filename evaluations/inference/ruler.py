"""Run local RULER JSONL tasks with OpenToMe KV-compression policies."""

import argparse
import json
import random
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import torch

from .kv_utils import (
    add_compression_args,
    add_model_args,
    build_cache,
    cache_stats,
    load_model_and_tokenizer,
    middle_truncate,
    run_metadata,
    synchronize,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    add_compression_args(parser)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Prediction JSONL path (default: work_dirs/ruler/"
            "<model>/<method>/<max-capacity-prompt>/predictions.jsonl)"
        ),
    )
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample", choices=("first", "random"), default="first")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_output_path(args):
    return (
        REPO_ROOT
        / "work_dirs"
        / "ruler"
        / Path(args.model_path).name
        / args.method
        / str(args.max_capacity_prompt)
        / "predictions.jsonl"
    )


def load_records(path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "input" not in record or "outputs" not in record:
                raise ValueError(f"{path}:{line_number} requires input and outputs fields")
            records.append(record)
    return records


def ruler_string_match(prediction, answers):
    if isinstance(answers, str):
        answers = [answers]
    if not answers:
        return 0.0
    prediction = prediction.lower()
    return sum(str(answer).lower() in prediction for answer in answers) / len(answers)


@torch.inference_mode()
def evaluate_record(model, tokenizer, record, args, device):
    prompt = record["input"]
    if args.chat_template and getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    model_limit = args.max_context_length or int(
        getattr(model.config, "max_position_embeddings", input_ids.shape[-1])
    )
    input_ids = middle_truncate(input_ids, max(1, model_limit - args.max_new_tokens)).to(device)
    cache = build_cache(args, model)
    synchronize(device)
    start = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        past_key_values=cache,
        use_cache=True,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    synchronize(device)
    elapsed = time.perf_counter() - start
    prediction = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
    return {
        "index": record.get("index"),
        "pred": prediction,
        "answers": record["outputs"],
        "length": record.get("length"),
        "input_tokens": input_ids.shape[-1],
        "generated_tokens": output.shape[-1] - input_ids.shape[-1],
        "string_match": ruler_string_match(prediction, record["outputs"]),
        "elapsed_seconds": elapsed,
        **cache_stats(cache),
    }


def main():
    args = parse_args()
    if args.output is None:
        args.output = default_output_path(args)
    torch.manual_seed(args.seed)
    metadata = run_metadata(args)
    records = load_records(args.data_file)
    if args.max_samples is not None and args.sample == "random":
        records = random.Random(args.seed).sample(records, min(args.max_samples, len(records)))
    elif args.max_samples is not None:
        records = records[:args.max_samples]
    model, tokenizer, device = load_model_and_tokenizer(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    completed = 0
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            completed = sum(1 for line in handle if line.strip())
    scores = []
    with args.output.open("a", encoding="utf-8") as handle:
        for record in records[completed:]:
            result = evaluate_record(model, tokenizer, record, args, device)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            scores.append(result["string_match"])
            print(json.dumps(result, ensure_ascii=False))
    summary = {
        "metadata": metadata,
        "records_total": len(records),
        "records_resumed": completed,
        "records_written": len(scores),
        "string_match_percent_new_records": round(sum(scores) / len(scores) * 100, 2) if scores else None,
    }
    write_json(args.output.with_suffix(".summary.json"), summary)


if __name__ == "__main__":
    main()
