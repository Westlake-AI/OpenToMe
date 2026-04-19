import math
import time
import torch
import datasets
import os
import glob
from loguru import logger
import torch.distributed as dist

from .training_utils import batch_fn


@torch.no_grad()
def evaluate_model(model, tokenizer, pad_idx, global_rank, world_size, device, args):
    _time = time.time()

    if os.path.isdir(args.data_path):
        # 本地目录：优先查找 c4-validation.*.json.gz
        val_files = sorted(glob.glob(os.path.join(args.data_path, "c4-validation.*.json.gz")))
        if val_files:
            logger.info(f"Loading validation data from {len(val_files)} local validation files")
            val_data = datasets.load_dataset(
                "json", data_files=val_files, split="train", streaming=True
            )
        else:
            # 没有本地 validation 文件，回退到 HuggingFace
            logger.warning(f"No c4-validation files found in {args.data_path}, falling back to HuggingFace allenai/c4")
            val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)
    else:
        val_data = datasets.load_dataset(args.data_path, args.data_name, split="validation", streaming=True)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")
    batch_size = args.batch_size

    # Issue-2 fix: removed split_dataset_by_node so evaluation is independent of world_size.
    # All ranks evaluate the same full validation set; results are identical across ranks.

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_tokens = torch.tensor(0.0).to(device)  # Issue-1: track token count for weighted average
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100

        n_tokens = (labels != -100).sum()
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach() * n_tokens  # Issue-1: accumulate token-weighted loss
        total_tokens += n_tokens

        # Issue-3 fix: removed * world_size; count only this rank's actual tokens
        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item()

    # Issue-1 fix: token-weighted average across all ranks
    # All ranks see the same data (no split), so values are identical; gather for synchronization
    if not args.single_gpu:
        gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
        gathered_tokens = [torch.zeros_like(total_tokens) for _ in range(world_size)]
        dist.all_gather(gathered_losses, total_loss)
        dist.all_gather(gathered_tokens, total_tokens)
        # All ranks computed the same data, so simple average equals any single rank's value
        total_loss_val = sum(l.item() for l in gathered_losses) / world_size
        total_tokens_val = sum(t.item() for t in gathered_tokens) / world_size
    else:
        total_loss_val = total_loss.item()
        total_tokens_val = total_tokens.item()

    avg_loss = total_loss_val / total_tokens_val
    perplexity = math.exp(avg_loss)

    return avg_loss, evaluated_on_tokens, perplexity