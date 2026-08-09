import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DNA_TO_ID = {
    "A": 7,
    "C": 8,
    "G": 9,
    "T": 10,
    "N": 11,
}
UNK_ID = 6


def open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path)


def encode_dna(sequence: str) -> list[int]:
    return [DNA_TO_ID.get(ch, UNK_ID) for ch in "".join(sequence.split()).upper()]


def resolve_torch_dtype(dtype: str, device: str):
    import torch

    dtype = dtype.lower()
    if dtype == "auto":
        if not str(device).startswith("cuda"):
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
    }
    dtype = aliases.get(dtype, dtype)
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"Unsupported dtype {dtype!r}") from exc


def load_hyenadna_causal_lm(model_dir: str | Path, device: str, dtype: str = "auto"):
    import opentome.models.hyena  # noqa: F401 - registers model_type=hyenadna
    import opentome.models.transformer  # noqa: F401 - registers model_type=transformer
    from transformers import AutoModelForCausalLM

    model_dir = Path(model_dir)
    weight_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if not any((model_dir / name).exists() for name in weight_files):
        expected = ", ".join(weight_files)
        raise FileNotFoundError(
            f"Expected HF-format model weights under {model_dir}; looked for: {expected}"
        )

    torch_dtype = resolve_torch_dtype(dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch_dtype,
    )
    return model.to(device=device, dtype=torch_dtype).eval(), model.config.to_dict()


def read_bed(
    bed_path: str | Path,
    split: str,
    limit: int | None = None,
) -> list[tuple[str, int, int]]:
    intervals = []
    with open(bed_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start, end, item_split, *_ = line.rstrip("\n").split("\t")
            if item_split != split:
                continue
            intervals.append((chrom, int(start), int(end)))
            if limit is not None and len(intervals) >= limit:
                break
    if not intervals:
        raise ValueError(f"No intervals for split={split!r} in {bed_path}")
    return intervals


def group_intervals(
    intervals: Iterable[tuple[str, int, int]]
) -> dict[str, list[tuple[int, int]]]:
    grouped = defaultdict(list)
    for chrom, start, end in intervals:
        grouped[chrom].append((start, end))
    return dict(grouped)


def iter_fasta_windows(
    fasta_path: str | Path,
    intervals: Iterable[tuple[str, int, int]],
    max_length: int,
) -> Iterator[tuple[str, str]]:
    grouped = group_intervals(intervals)
    current_chrom = None
    current_parts: list[str] = []

    def emit_chrom(chrom: str | None, parts: list[str]):
        if chrom not in grouped or not parts:
            return
        chromosome = "".join(parts).upper()
        for start, end in grouped[chrom]:
            seq = chromosome[start:end]
            if max_length > 0:
                seq = seq[:max_length]
            if len(seq) >= 2:
                yield chrom, seq

    with open_text(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                yield from emit_chrom(current_chrom, current_parts)
                current_chrom = line[1:].split()[0]
                current_parts = []
                continue
            if current_chrom in grouped:
                current_parts.append(line)
        yield from emit_chrom(current_chrom, current_parts)


def score_sequence(model, sequence: str, device: str) -> tuple[float, int]:
    import torch
    import torch.nn.functional as F

    ids = encode_dna(sequence)
    if len(ids) < 2:
        return 0.0, 0
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.inference_mode():
        outputs = model(input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs

    shift_logits = logits[:, :-1].float().contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="sum",
    )
    return float(loss.item()), int(shift_labels.numel())


def evaluate(args) -> dict[str, float | int | str]:
    import torch
    from tqdm import tqdm

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_hyenadna_causal_lm(args.model_dir, device, args.dtype)

    max_model_len = int(config.get("max_seq_len", args.max_length + 2)) - 2
    max_length = min(args.max_length, max_model_len) if args.max_length > 0 else max_model_len
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None
    intervals = read_bed(args.bed, args.split, max_samples)

    total_nll = 0.0
    total_tokens = 0
    total_sequences = 0

    windows = iter_fasta_windows(args.fasta, intervals, max_length=max_length)
    for _chrom, seq in tqdm(windows, total=len(intervals), desc=f"HyenaDNA PPL {args.split}"):
        nll, ntokens = score_sequence(model, seq, device)
        if ntokens == 0:
            continue
        total_nll += nll
        total_tokens += ntokens
        total_sequences += 1

    if total_tokens == 0:
        raise RuntimeError("No tokens were evaluated")
    mean_nll = total_nll / total_tokens
    return {
        "model_dir": str(args.model_dir),
        "fasta": str(args.fasta),
        "bed": str(args.bed),
        "split": args.split,
        "max_length": max_length,
        "sequences": total_sequences,
        "tokens": total_tokens,
        "nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run HF-format HyenaDNA inference on hg38 bed windows and report next-token PPL."
    )
    parser.add_argument("--model_dir", default="models/hyenadna-small-32k-seqlen-hf")
    parser.add_argument("--fasta", default="data/hg38/hg38.ml.fa.gz")
    parser.add_argument("--bed", default="data/hg38/human-sequences.bed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Inference dtype: auto, bfloat16/bf16, float16/fp16, or float32/fp32.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = evaluate(args)
    print(json.dumps(results, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
