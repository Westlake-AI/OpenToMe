"""
长序列 Scaling 基准脚本
对比：
- Baseline: 标准 Transformer (全局注意力，O(L^2))，实现见 `model_flashattn.py`
- MergeNet: 线性/近线性复杂度 (Unfold Storage + Sliding Window 1)，实现见 `model.py`

功能：
- 在一组序列长度 L 上（默认 2k~64k）测 Training Throughput (tokens/s) 与显存峰值
- 使用 DDP（默认 8×A100），每个 rank 执行同样的长度 sweep
- 数据使用随机张量，避免 IO 干扰

运行示例（8 卡）：
torchrun --nproc_per_node=8 evaluations/seq_length_scaling_ddp.py \
  --lengths 2048,4096,8192,16384,32768,65536 \
  --batch-size 1 --grad-accum 1 \
  --steps 20 --warmup 5 \
  --dtype bf16 \
  --mode both \
  --lambda-local 4 \
  --output results/seq_scaling_ddp.csv

说明：
- 模型配置默认与 `HybridToMeModel` 的 base 配置一致，可通过参数修改。
- 序列长度通过将 patch_size 设为 1，并将 img_size 调整为 ceil(sqrt(L)) 来近似得到 L 个 token。
- 训练吞吐量按 “总 token 数 / 实测时间” 计算，其中总 token 数 = (img_size^2 + 1(cls)) * batch_size * world_size * grad_accum * steps。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.amp import autocast, GradScaler  # type: ignore
except Exception:  # pragma: no cover
    from torch.cuda.amp import autocast, GradScaler  # type: ignore

# 将 repo 根目录加入 sys.path，便于脚本从 evaluations/ 下直接运行
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from opentome.models.mergenet.model import HybridToMeModel  # noqa: E402
from opentome.models.mergenet.model_flashattn import FlashAttentionModel  # noqa: E402


def parse_lengths(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x]


def init_distributed(backend: str = "nccl") -> int:
    if dist.is_initialized():
        return dist.get_rank()
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def build_mergenet(args, img_size: int) -> torch.nn.Module:
    model = HybridToMeModel(
        arch=args.arch,
        img_size=img_size,
        patch_size=args.patch_size,
        dtem_feat_dim=args.dtem_feat_dim,
        dtem_window_size=args.dtem_window_size,
        tome_window_size=args.tome_window_size,
        lambda_local=args.lambda_local,
        total_merge_latent=args.total_merge_latent,
        dtem_t=args.dtem_t,
        use_softkmax=args.use_softkmax,
        local_block_window=args.local_block_window,
    )
    # 论文中的 DTEM 温度默认值（保持与训练脚本一致）
    model.local.vit._tome_info["k2"] = args.k2
    model.local.vit._tome_info["tau1"] = args.tau1
    model.local.vit._tome_info["tau2"] = args.tau2
    return model


def build_baseline(args, img_size: int) -> torch.nn.Module:
    depth = args.local_depth + args.latent_depth
    model = FlashAttentionModel(
        img_size=img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        depth=depth,
        num_classes=args.num_classes,
        h=args.local_window,
    )
    return model


def benchmark_one_model(
    kind: str,
    lengths: List[int],
    args,
    device: torch.device,
    world_size: int,
    local_rank: int,
    mergenet_mode: str | None = None,
) -> List[Dict]:
    results: List[Dict] = []
    torch.manual_seed(args.seed + local_rank)

    for L in lengths:
        err_flag = torch.zeros(1, device=device, dtype=torch.int32)
        # 通过设置 patch_size=1，将 seq_len 近似映射为 H*W token
        img_size = int(math.ceil(math.sqrt(L)))
        tokens_per_sample = img_size * img_size + 1  # +1 for cls token

        try:
            torch.cuda.empty_cache()
            if kind == "mergenet":
                model = build_mergenet(args, img_size)
            else:
                model = build_baseline(args, img_size)

            model.to(device)
            model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

            mode = mergenet_mode if kind == "mergenet" and mergenet_mode is not None else "train"
            batch = args.batch_size
            use_amp = args.dtype in ("bf16", "fp16")
            amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
            scaler = GradScaler(device.type) if (args.dtype == "fp16" and mode == "train") else None
            input_dtype = amp_dtype if use_amp else torch.float32
            inputs = torch.randn(batch, 3, img_size, img_size, device=device, dtype=input_dtype)
            targets = torch.randint(0, args.num_classes, (batch,), device=device)

            if mode == "train":
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                criterion = torch.nn.CrossEntropyLoss().to(device)
                model.train()
            else:
                model.eval()
                optimizer = None
                criterion = None

            torch.cuda.reset_peak_memory_stats(device)

            # warmup
            for _ in range(args.warmup):
                if mode == "train":
                    optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with autocast(device_type="cuda", dtype=amp_dtype):
                        logits, _ = model(inputs)
                        loss = None if criterion is None else criterion(logits, targets) / args.grad_accum
                    if mode == "train":
                        if loss is not None:
                            if scaler is not None:
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                else:
                    logits, _ = model(inputs)
                    loss = None if criterion is None else criterion(logits, targets) / args.grad_accum
                    if mode == "train" and loss is not None:
                        loss.backward()
                if mode == "train":
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

            torch.cuda.synchronize(device)
            start = time.time()
            for _ in range(args.steps):
                if mode == "train":
                    optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with autocast(device_type="cuda", dtype=amp_dtype):
                        logits, _ = model(inputs)
                        loss = None if criterion is None else criterion(logits, targets) / args.grad_accum
                    if mode == "train" and loss is not None:
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                else:
                    logits, _ = model(inputs)
                    loss = None if criterion is None else criterion(logits, targets) / args.grad_accum
                    if mode == "train" and loss is not None:
                        loss.backward()
                if mode == "train":
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.time() - start

            # 聚合最大显存
            mem_tensor = torch.tensor(
                [torch.cuda.max_memory_reserved(device)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(mem_tensor, op=dist.ReduceOp.MAX)
            peak_mem_gb = float(mem_tensor.item() / (1024 ** 3))

            total_tokens = tokens_per_sample * batch * world_size * args.grad_accum * args.steps
            throughput = total_tokens / elapsed if elapsed > 0 else 0.0

            if local_rank == 0:
                results.append(
                    {
                        "model": kind,
                        "seq_len": L,
                        "effective_tokens": tokens_per_sample,
                        "throughput_tokens_per_s": throughput,
                        "peak_mem_gb": peak_mem_gb,
                        "batch": batch,
                        "grad_accum": args.grad_accum,
                        "steps": args.steps,
                        "dtype": args.dtype,
                        "world_size": world_size,
                        "img_size": img_size,
                        "lambda_local": args.lambda_local if kind == "mergenet" else None,
                    }
                )
                print(
                    f"[{kind}] L≈{L} (img {img_size}x{img_size}, tokens {tokens_per_sample}) "
                    f"throughput {throughput/1e6:.3f}M tok/s, peak_mem {peak_mem_gb:.2f} GB"
                )

            # 清理
            del model, optimizer, criterion, scaler, inputs, targets
            torch.cuda.empty_cache()

        except RuntimeError as e:
            err_flag.fill_(1)
            if "out of memory" in str(e).lower():
                if local_rank == 0:
                    results.append(
                        {
                            "model": kind,
                            "seq_len": L,
                            "effective_tokens": tokens_per_sample,
                            "throughput_tokens_per_s": 0.0,
                            "peak_mem_gb": float("nan"),
                            "batch": args.batch_size,
                            "grad_accum": args.grad_accum,
                            "steps": args.steps,
                            "dtype": args.dtype,
                            "world_size": world_size,
                            "img_size": img_size,
                            "lambda_local": args.lambda_local if kind == "mergenet" else None,
                            "oom": True,
                            "error": str(e),
                        "mode": mode,
                        }
                    )
                    print(f"[{kind}] L={L} OOM，跳过 (img {img_size}x{img_size})")
                torch.cuda.empty_cache()
            else:
                if local_rank == 0:
                    results.append(
                        {
                            "model": kind,
                            "seq_len": L,
                            "effective_tokens": tokens_per_sample,
                            "throughput_tokens_per_s": 0.0,
                            "peak_mem_gb": float("nan"),
                            "batch": args.batch_size,
                            "grad_accum": args.grad_accum,
                            "steps": args.steps,
                            "dtype": args.dtype,
                            "world_size": world_size,
                            "img_size": img_size,
                            "lambda_local": args.lambda_local if kind == "mergenet" else None,
                            "oom": False,
                            "error": str(e),
                        "mode": mode,
                        }
                    )
                    print(f"[{kind}] L={L} 遇到错误，跳过: {e}")
                torch.cuda.empty_cache()

        # 让所有 rank 同步状态，避免个别 rank 抛异常导致不一致
        dist.all_reduce(err_flag, op=dist.ReduceOp.MAX)
        if err_flag.item() > 0:
            continue

    return results


def save_results(path: str, rows: List[Dict]):
    if not rows:
        return
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    keys = [
        "model",
        "seq_len",
        "effective_tokens",
        "throughput_tokens_per_s",
        "peak_mem_gb",
        "batch",
        "grad_accum",
        "steps",
        "dtype",
        "world_size",
        "img_size",
        "lambda_local",
        "oom",
    ]
    # 确保所有 key 存在
    for row in rows:
        for k in keys:
            row.setdefault(k, None)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Rank0] 结果已写入 {path}")


def get_args():
    parser = argparse.ArgumentParser(description="Sequence Length Scaling Benchmark (DDP)")
    parser.add_argument("--lengths", type=str, default="2048,4096,8192,16384,32768,65536")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3, help="warmup optimizer steps")
    parser.add_argument("--steps", type=int, default=10, help="bench optimizer steps")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mode", type=str, default="both", choices=["both", "mergenet", "baseline"])
    parser.add_argument("--mergenet-mode", type=str, default="infer", choices=["infer", "train"],
                        help="为避免 DTEM 训练态异常，默认只做前向推理测速；如需训练吞吐改为 train")
    parser.add_argument("--output", type=str, default="results/seq_scaling_ddp.csv")
    parser.add_argument("--seed", type=int, default=42)
    # 模型公共参数
    parser.add_argument("--patch-size", type=int, default=1, help="使用 patch_size=1 将序列映射为 H*W token")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    # baseline 配置
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--local-depth", type=int, default=4, help="baseline depth 前半与 MergeNet 对齐")
    parser.add_argument("--latent-depth", type=int, default=12, help="baseline depth 后半与 MergeNet 对齐")
    parser.add_argument("--local-window", type=int, default=None, help="baseline 可选局部窗口；None=全局注意力")
    # MergeNet 配置
    parser.add_argument("--arch", type=str, default="base", choices=["base", "small"])
    parser.add_argument("--lambda-local", type=float, default=4.0)
    parser.add_argument("--total-merge-latent", type=int, default=4)
    parser.add_argument("--dtem-window-size", type=int, default=1, help="Sliding Window=1 对齐论文假设")
    parser.add_argument("--tome-window-size", type=int, default=None)
    parser.add_argument("--dtem-t", type=int, default=1)
    parser.add_argument("--use-softkmax", action="store_true")
    parser.add_argument("--dtem-feat-dim", type=int, default=64)
    parser.add_argument("--local-block-window", type=int, default=16)
    parser.add_argument("--k2", type=float, default=4.0)
    parser.add_argument("--tau1", type=float, default=1.0)
    parser.add_argument("--tau2", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = get_args()
    lengths = parse_lengths(args.lengths)
    local_rank = init_distributed()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    all_results: List[Dict] = []
    if args.mode in ("mergenet", "both"):
        res = benchmark_one_model("mergenet", lengths, args, device, world_size, local_rank, mergenet_mode=args.mergenet_mode)
        if local_rank == 0:
            all_results.extend(res)

    if args.mode in ("baseline", "both"):
        res = benchmark_one_model("baseline", lengths, args, device, world_size, local_rank)
        if local_rank == 0:
            all_results.extend(res)

    if local_rank == 0:
        save_results(args.output, all_results)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# cd /liziqing/yuhao/yukai/OpenToMe
# torchrun --nproc_per_node=8 evaluations/seq_length_scaling_ddp.py --lengths 2048,4096,8192,16384,32768,65536 --batch-size 1 --grad-accum 1 --steps 20 --warmup 5 --dtype bf16 --mode both --lambda-local 4 --output results/seq_scaling_ddp.csv