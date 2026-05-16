#!/usr/bin/env python3
"""
Run one training epoch (same recipe as c100_a0.sh) and report ToME token counts.

  cd /path/to/OpenToMe && PYTHONPATH=. python trainer/classification/measure_tome_token_compression_one_epoch.py

Does not import in1k_trainer.py (avoids heavy torchvision/onnx import chain in some envs).
"""
from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from collections import Counter

import torch

os.environ.setdefault("OPENTOME_MERGENET_IMPL", "tome")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import opentome.models.mergenet.model_a0  # noqa: F401  # registers tomevit_small_cls in timm


def build_train_args(data_dir: str, debug_subset: int) -> Namespace:
    """Mirror c100_a0.sh + timm create_loader defaults used by build_dataset."""
    return Namespace(
        data_dir=data_dir,
        dataset="CIFAR100",
        train_split="train",
        val_split="val",
        dataset_download=False,
        debug_subset=debug_subset,
        num_classes=100,
        batch_size=50,
        validation_batch_size=None,
        prefetcher=False,
        no_aug=False,
        reprob=0.25,
        remode="pixel",
        recount=1,
        resplit=False,
        scale=[0.08, 1.0],
        ratio=[3.0 / 4.0, 4.0 / 3.0],
        hflip=0.5,
        vflip=0.0,
        color_jitter=0.4,
        aa="rand-m9-mstd0.5-inc1",
        aug_repeats=0,
        aug_splits=0,
        train_interpolation="random",
        workers=8,
        distributed=False,
        pin_mem=True,
        use_multi_epochs_loader=False,
        worker_seeding="all",
        img_size=224,
        patch_size=16,
        crop_pct=0.90,
        interpolation="bicubic",
        mean=None,
        std=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/liziqing/yuhao/yukai/data")
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--debug_subset", type=int, default=0)
    cli = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (matches training setup).")
    torch.cuda.set_device(0)

    args = build_train_args(cli.data_dir, cli.debug_subset)

    from timm.models import create_model, model_parameters
    from timm.data import resolve_data_config
    from timm.optim import create_optimizer_v2, optimizer_kwargs
    from timm.loss import SoftTargetCrossEntropy
    from timm.utils import NativeScaler
    from timm.data import Mixup

    from opentome.utils.dataset_loader import build_dataset

    model_kwargs = {
        "pretrained": False,
        "num_classes": 100,
        "img_size": 224,
        "patch_size": 16,
        "dtem_window_size": 7,
        "dtem_r": 2,
        "dtem_t": 1,
        "dtem_feat_dim": None,
        "lambda_local": 4.0,
        "total_merge_latent": 0,
        "use_softkmax": False,
        "local_block_window": 16,
        "tome_window_size": None,
        "tome_use_naive_local": False,
        "swa_size": None,
        "pretrained_type": "vit",
        "load_full_pretrained": True,
        "freeze_local_encoder": False,
    }
    model = create_model("tomevit_small_cls", **model_kwargs).cuda()

    full_cfg = {
        **vars(args),
        "model": "tomevit_small_cls",
        "amp": True,
        "lr": 1e-3,
        "lr_local": 1e-3,
        "opt": "adamw",
        "weight_decay": 0.05,
        "clip_grad": 1.0,
        "clip_mode": "norm",
        "sched": "cosine",
        "warmup_epochs": 5,
        "epochs": 1,
        "mixup": 0.8,
        "cutmix": 1.0,
        "cutmix_minmax": None,
        "mixup_prob": 1.0,
        "mixup_switch_prob": 0.5,
        "mixup_mode": "batch",
        "smoothing": 0.1,
    }
    data_config = resolve_data_config(full_cfg, model=model, verbose=True)

    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=100,
    )

    loader_train, _ = build_dataset(args, data_config, collate_fn=None, num_aug_splits=0)

    opt_args = Namespace(
        **full_cfg,
        opt_eps=None,
        opt_betas=None,
        momentum=0.9,
        opt_args=None,
    )
    # c100_a0 中 lr == lr_local；避免依赖 torchtitan
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=opt_args))

    amp_autocast = lambda: torch.amp.autocast(device_type="cuda", enabled=True)
    loss_scaler = NativeScaler()
    train_loss_fn = SoftTargetCrossEntropy().cuda()

    num_patches = (224 // 16) ** 2
    n0 = num_patches + 1
    total_merge_local = int(num_patches * (4.0 - 1.0) / 4.0)

    # 注：aux["token_counts_local"] 仅在 ToME 循环内 append，首元素是「第一次 merge 之后」的长度
    #（例如 197->161），不是 patch_embed 后的 197。
    print("=== 理论值（与 ToMELocalEncoder 一致）===")
    print(f"  num_patches={num_patches}, 进入 ToME merge 前 seq_len ≈ {n0} (含 CLS)")
    print(f"  lambda_local=4.0 -> total_merge_local={total_merge_local}")
    print(f"  合并后 token 数 ≈ {n0 - total_merge_local}")

    local_pattern_counter: Counter = Counter()
    latent_pattern_counter: Counter = Counter()
    batches_done = 0

    model.train()
    for batch_idx, (input, target) in enumerate(loader_train):
        if cli.max_batches is not None and batch_idx >= cli.max_batches:
            break
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        input, target = mixup_fn(input, target)

        optimizer.zero_grad(set_to_none=True)
        with amp_autocast():
            out = model(input)
            if isinstance(out, (tuple, list)):
                logits, aux = out
            else:
                logits, aux = out, {}
            loss = train_loss_fn(logits, target)

        loss_scaler(
            loss,
            optimizer,
            clip_grad=1.0,
            clip_mode="norm",
            parameters=model_parameters(model, exclude_head=False),
        )

        tc_l = aux.get("token_counts_local")
        if tc_l is not None:
            local_pattern_counter[tuple(int(x) for x in tc_l)] += 1

        latent_counts = None
        if getattr(model, "latent", None) is not None and hasattr(model.latent, "vit"):
            info = getattr(model.latent.vit, "_tome_info", None)
            if info is not None:
                latent_counts = info.get("token_counts_latent")

        if latent_counts:
            latent_pattern_counter[tuple(int(x) for x in latent_counts)] += 1

        batches_done += 1

    print("\n=== 实测（训练模式，本脚本跑的 batch 数）===")
    print(f"  batches={batches_done}")
    if not local_pattern_counter:
        print("  未收集到 token_counts_local")
    else:
        print("  token_counts_local 序列 -> 出现次数：")
        for pattern, cnt in local_pattern_counter.most_common():
            print(f"    {list(pattern)}  ->  {cnt} batches")
        last_local = local_pattern_counter.most_common(1)[0][0]
        n_after_local = last_local[-1] if last_local else n0
        print(
            f"\n  Local：初始≈{n0} -> 最终 {n_after_local} tokens | "
            f"保留比例 {n_after_local / n0:.4f} | 压缩比例 {1.0 - n_after_local / n0:.4f}"
        )

    if not latent_pattern_counter:
        print("\n  未收集到 token_counts_latent")
    else:
        print("\n  token_counts_latent（逐 block）-> 出现次数：")
        for pattern, cnt in latent_pattern_counter.most_common():
            print(f"    {list(pattern)}  ->  {cnt} batches")
        lp = latent_pattern_counter.most_common(1)[0][0]
        if lp:
            z0, z1 = lp[0], lp[-1]
            print(
                f"\n  Latent：首记录 {z0} -> 末记录 {z1} "
                f"（total_merge_latent=0 时应全程不变）"
            )


if __name__ == "__main__":
    main()
