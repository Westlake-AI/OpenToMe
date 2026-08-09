import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentome.tokenizer import DNATokenizer


def import_backbone(backbone: str):
    if "gated_deltanet" in backbone:
        import opentome.models.gated_deltanet  # noqa: F401
    elif "blt" in backbone:
        import opentome.models.blt  # noqa: F401
    elif "delta_net" in backbone:
        import opentome.models.delta_net  # noqa: F401
    elif "gla" in backbone:
        import opentome.models.gla  # noqa: F401
    elif "gsa" in backbone:
        import opentome.models.gsa  # noqa: F401
    elif "transformer++" in backbone or "transformer" in backbone:
        import opentome.models.transformer  # noqa: F401
    elif "mergenet" in backbone:
        import opentome.models.mergenet_nlp  # noqa: F401
    elif "hnet" in backbone:
        import opentome.models.hnet  # noqa: F401
    elif "hyena" in backbone or "hyenadna" in backbone:
        import opentome.models.hyena  # noqa: F401
    else:
        raise ValueError(
            f"Unsupported backbone={backbone!r}. Use transformer++, delta_net, "
            "gated_deltanet, blt, gla, gsa, mergenet, hnet, or hyenadna."
        )


def open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path)


def read_bed(path: str | Path, split: str) -> list[tuple[str, int, int]]:
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start, end, item_split, *_ = line.rstrip("\n").split("\t")
            if item_split == split:
                rows.append((chrom, int(start), int(end)))
    if not rows:
        raise ValueError(f"No intervals for split={split!r} in {path}")
    return rows


def load_fasta(path: str | Path, chroms: set[str]) -> dict[str, str]:
    genome = {}
    current = None
    parts = []

    def flush():
        if current in chroms and parts:
            genome[current] = "".join(parts).upper()

    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current = line[1:].split()[0]
                parts = []
            elif current in chroms:
                parts.append(line)
        flush()
    missing = chroms.difference(genome)
    if missing:
        raise ValueError(f"Missing chromosomes in FASTA: {sorted(missing)[:8]}")
    return genome


class HG38CausalDataset(Dataset):
    def __init__(
        self,
        fasta: str | Path,
        bed: str | Path,
        split: str,
        seq_len: int,
        tokenizer: DNATokenizer,
        pad_token: str = "N",
    ):
        self.intervals = read_bed(bed, split)
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        chroms = {chrom for chrom, _, _ in self.intervals}
        self.genome = load_fasta(fasta, chroms)

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, idx: int):
        chrom, start, end = self.intervals[idx]
        chrom_seq = self.genome[chrom]
        seq = chrom_seq[start:end]
        if len(seq) >= self.seq_len:
            seq = seq[: self.seq_len]
        else:
            seq = seq + self.pad_token * (self.seq_len - len(seq))
        ids = self.tokenizer.encode(seq, add_special_tokens=False)
        input_ids = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_distributed(enabled: bool):
    if enabled:
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int, rank: int):
    seed = seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_length(value: str) -> int:
    value = str(value).strip().lower().replace("_", "")
    multipliers = {"k": 1024, "m": 1024 * 1024}
    if value[-1:] in multipliers:
        return int(float(value[:-1]) * multipliers[value[-1]])
    return int(value)


def parse_seq_len_schedule(spec: str | None, default_seq_len: int) -> list[tuple[int, int]]:
    if not spec:
        return [(0, default_seq_len)]

    schedule: list[tuple[int, int]] = []
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "--seq_len_schedule entries must be boundary:length, "
                "for example 0:32k,3:64k,6:1m"
            )
        boundary, seq_len = item.split(":", 1)
        boundary = int(boundary.strip())
        seq_len = parse_length(seq_len)
        if boundary < 0:
            raise ValueError(f"Schedule boundary must be >= 0, got {boundary}")
        if seq_len <= 1:
            raise ValueError(f"Schedule seq_len must be > 1, got {seq_len}")
        schedule.append((boundary, seq_len))

    if not schedule:
        raise ValueError("--seq_len_schedule did not contain any valid entries")
    schedule.sort(key=lambda item: item[0])
    if schedule[0][0] != 0:
        schedule.insert(0, (0, default_seq_len))

    deduped: list[tuple[int, int]] = []
    for boundary, seq_len in schedule:
        if deduped and deduped[-1][0] == boundary:
            deduped[-1] = (boundary, seq_len)
        else:
            deduped.append((boundary, seq_len))
    return deduped


def seq_len_for_position(schedule: list[tuple[int, int]], position: int) -> int:
    active = schedule[0][1]
    for boundary, seq_len in schedule:
        if position < boundary:
            break
        active = seq_len
    return active


def iter_config_tree(config):
    yield config
    for name in ("patcher_config", "encoder_config", "decoder_config", "global_config"):
        sub_config = getattr(config, name, None)
        if sub_config is not None:
            yield sub_config


def set_max_position_embeddings(config, model_seq_len: int):
    for item in iter_config_tree(config):
        if hasattr(item, "max_position_embeddings") and item.max_position_embeddings < model_seq_len:
            item.max_position_embeddings = model_seq_len
        if hasattr(item, "max_seq_len") and item.max_seq_len < model_seq_len:
            item.max_seq_len = model_seq_len


def set_attention_implementation(config, attn_implementation: str | None):
    if attn_implementation is None:
        return
    for item in iter_config_tree(config):
        item._attn_implementation = attn_implementation


def build_model(args, tokenizer: DNATokenizer, device: torch.device):
    import_backbone(args.backbone)
    config = AutoConfig.from_pretrained(args.model_config)
    config.vocab_size = tokenizer.get_vocab_size()
    model_seq_len = getattr(args, "model_seq_len", args.seq_len)
    set_max_position_embeddings(config, model_seq_len)
    set_attention_implementation(config, args.attn_implementation)
    if hasattr(config, "use_cache"):
        config.use_cache = False
    if hasattr(config, "pad_token_id"):
        config.pad_token_id = tokenizer.pad_token_id
    if hasattr(config, "bos_token_id"):
        config.bos_token_id = tokenizer.bos_token_id
    if hasattr(config, "eos_token_id"):
        config.eos_token_id = tokenizer.eos_token_id
    if args.disable_fused_loss:
        for key in ("fuse_linear_cross_entropy", "fuse_cross_entropy"):
            if hasattr(config, key):
                setattr(config, key, False)
    model = AutoModelForCausalLM.from_config(config)
    model.to(device)
    return model, config


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, tokenizer: DNATokenizer, output_dir: str | Path, step: int, config):
    output_dir = Path(output_dir) / f"step-{step}"
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "dna_train_config.json", "w") as f:
        json.dump({"step": step, "model_config": config.to_dict()}, f, indent=2)


def train(args):
    distributed, rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, rank)

    tokenizer = DNATokenizer(add_bos=False, add_eos=False, padding_side="right")
    seq_len_schedule = parse_seq_len_schedule(args.seq_len_schedule, args.seq_len)
    args.model_seq_len = max(seq_len for _, seq_len in seq_len_schedule)
    initial_seq_len = seq_len_for_position(seq_len_schedule, 0)
    dataset = HG38CausalDataset(
        fasta=args.fasta,
        bed=args.bed,
        split=args.split,
        seq_len=initial_seq_len,
        tokenizer=tokenizer,
    )

    if args.global_batch_size is not None:
        args.grad_accum = max(1, math.ceil(args.global_batch_size / (world_size * args.batch_size)))
    effective_global_batch = world_size * args.batch_size * args.grad_accum
    updates_per_epoch = math.ceil(len(dataset) / effective_global_batch)
    if args.epochs is not None:
        total_steps = updates_per_epoch * args.epochs
    elif args.steps is not None:
        total_steps = args.steps
    else:
        raise ValueError("Set either --epochs or --steps.")
    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = max(1, math.ceil(total_steps * args.warmup_ratio)) if args.warmup_ratio > 0 else 0

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=False) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model, config = build_model(args, tokenizer, device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=args.find_unused_parameters)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps, args.min_lr_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "fp16")
    autocast_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    use_autocast = args.mixed_precision in {"fp16", "bf16"} and device.type == "cuda"

    if is_main(rank):
        schedule_desc = ", ".join(f"{boundary}:{seq_len}" for boundary, seq_len in seq_len_schedule)
        print(f"Dataset split={args.split}, samples={len(dataset)}, initial_seq_len={initial_seq_len}, model_seq_len={args.model_seq_len}")
        print(f"Seq len schedule unit={args.seq_len_schedule_unit}, schedule={schedule_desc}")
        print(f"Backbone={args.backbone}, config={args.model_config}")
        print(f"World size={world_size}, batch_size/device={args.batch_size}, grad_accum={args.grad_accum}")
        print(f"Effective global batch={effective_global_batch}, epochs={args.epochs}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    def optimizer_step():
        if args.max_grad_norm > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    def log_step(step: int, running_loss: float, running_tokens: int, start_time: float):
        elapsed = max(time.time() - start_time, 1e-6)
        avg_loss = running_loss / args.log_every
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        toks_per_sec = running_tokens * world_size / elapsed
        lr = scheduler.get_last_lr()[0]
        print(
            f"step={step} seq_len={dataset.seq_len} loss={avg_loss:.4f} "
            f"ppl={ppl:.4f} lr={lr:.3e} toks/s={toks_per_sec:.1f}",
            flush=True,
        )

    def maybe_update_seq_len(position: int, unit: str) -> bool:
        if args.seq_len_schedule_unit != unit:
            return False
        next_seq_len = seq_len_for_position(seq_len_schedule, position)
        if dataset.seq_len == next_seq_len:
            return False
        dataset.seq_len = next_seq_len
        if is_main(rank):
            print(f"seq_len changed to {next_seq_len} at {unit}={position}", flush=True)
        return True

    model.train()
    step = 0
    running_loss = 0.0
    running_tokens = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    if args.epochs is not None:
        for epoch in range(args.epochs):
            maybe_update_seq_len(epoch, "epoch")
            if sampler is not None:
                sampler.set_epoch(epoch)
            accum_count = 0
            accum_loss = 0.0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / args.grad_accum
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                accum_count += 1
                accum_loss += float(loss.detach().cpu())
                running_tokens += labels.numel()

                if accum_count == args.grad_accum:
                    optimizer_step()
                    step += 1
                    running_loss += accum_loss
                    accum_count = 0
                    accum_loss = 0.0
                    if is_main(rank) and step % args.log_every == 0:
                        log_step(step, running_loss, running_tokens, start_time)
                        running_loss = 0.0
                        running_tokens = 0
                        start_time = time.time()
                    if args.save_every > 0 and step % args.save_every == 0 and is_main(rank):
                        save_checkpoint(model, tokenizer, args.output_dir, step, config)

            if accum_count > 0:
                optimizer_step()
                step += 1
                running_loss += accum_loss
                if is_main(rank) and step % args.log_every == 0:
                    log_step(step, running_loss, running_tokens, start_time)
                    running_loss = 0.0
                    running_tokens = 0
                    start_time = time.time()
                if args.save_every > 0 and step % args.save_every == 0 and is_main(rank):
                    save_checkpoint(model, tokenizer, args.output_dir, step, config)
    else:
        epoch = 0
        data_iter = iter(dataloader)
        while step < total_steps:
            if maybe_update_seq_len(step, "step"):
                data_iter = iter(dataloader)
            if sampler is not None:
                sampler.set_epoch(epoch)
            accum_loss = 0.0
            for _ in range(args.grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / args.grad_accum
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                accum_loss += float(loss.detach().cpu())
                running_tokens += labels.numel()

            optimizer_step()
            step += 1
            running_loss += accum_loss
            if is_main(rank) and step % args.log_every == 0:
                log_step(step, running_loss, running_tokens, start_time)
                running_loss = 0.0
                running_tokens = 0
                start_time = time.time()
            if args.save_every > 0 and step % args.save_every == 0 and is_main(rank):
                save_checkpoint(model, tokenizer, args.output_dir, step, config)

    if is_main(rank):
        save_checkpoint(model, tokenizer, args.output_dir, step, config)
    cleanup_distributed(distributed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train OpenToMe causal LMs on HG38 with DNATokenizer.")
    parser.add_argument("--backbone", default="transformer++")
    parser.add_argument("--model_config", default="trainer/flame/configs/transformer_340M.json")
    parser.add_argument("--fasta", default="data/hg38/hg38.ml.fa.gz")
    parser.add_argument("--bed", default="data/hg38/human-sequences.bed")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="outputs/dna_train")
    parser.add_argument("--seq_len", type=parse_length, default=32768)
    parser.add_argument(
        "--seq_len_schedule",
        default=None,
        help=(
            "Comma-separated boundary:length curriculum, e.g. 0:32k,3:64k,6:1m. "
            "Boundaries are epochs by default, or optimizer steps with --seq_len_schedule_unit step."
        ),
    )
    parser.add_argument("--seq_len_schedule_unit", choices=["epoch", "step"], default="epoch")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--global_batch_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default=None)
    parser.add_argument("--disable_fused_loss", action="store_true")
    parser.add_argument("--find_unused_parameters", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
