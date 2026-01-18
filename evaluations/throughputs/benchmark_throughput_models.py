import argparse
import os
import time
import traceback
from typing import Optional, Tuple

import pandas as pd
import torch


def bench_model(model_name: str,
                img_size: int,
                patch_size: int,
                embed_dim: int,
                num_heads: int,
                mlp_ratio: float,
                local_depth: int,
                latent_depth: int,
                dtem_window_size: Optional[int],
                tome_window_size: Optional[int],
                total_merge_local: int,
                total_merge_latent: int,
                batch_size: int,
                warmup: int,
                iters: int,
                device: str,
                mode: str,
                suppress_oom: bool) -> Tuple[float, float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.startswith('cuda') and torch.cuda.is_available()

    try:
        x = torch.randn(batch_size, 3, img_size, img_size, device=device, requires_grad=(mode == 'fwd_bwd'))

        if model_name == 'flash':
            from opentome.models.model_flashattn import FlashAttentionModel
            depth = local_depth + latent_depth
            model = FlashAttentionModel(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                depth=depth,
                num_classes=10,
            ).to(device)
        elif model_name == 'hybrid':
            from opentome.models.mergenet.model import HybridToMeModel
            # Determine arch based on embed_dim and latent_depth
            # arch_zoo: small (latent_depth=8), s_ext (latent_depth=12), base (latent_depth=8)
            if embed_dim == 384:
                if latent_depth == 12:
                    arch = 's_ext'  # small_extend
                elif latent_depth == 8:
                    arch = 'small'
                else:
                    raise ValueError(f"Unsupported latent_depth {latent_depth} for embed_dim=384. Supported: 8 (small), 12 (s_ext)")
            elif embed_dim == 768:
                if latent_depth == 8:
                    arch = 'base'
                else:
                    raise ValueError(f"Unsupported latent_depth {latent_depth} for embed_dim=768. Supported: 8 (base)")
            else:
                raise ValueError(f"Unsupported embed_dim: {embed_dim}. Supported: 384 (small/s_ext), 768 (base)")
            
            # Calculate lambda_local from total_merge_local
            num_patches = (img_size // patch_size) ** 2
            if total_merge_local > 0:
                # lambda_local = num_patches / (num_patches - total_merge_local)
                lambda_local = num_patches / max(1, num_patches - total_merge_local)
            else:
                lambda_local = 2.0  # default
            
            model = HybridToMeModel(
                arch=arch,
                img_size=img_size,
                patch_size=patch_size,
                dtem_feat_dim=64,
                tome_use_naive_local=False,
                lambda_local=lambda_local,
                total_merge_latent=total_merge_latent,
                dtem_window_size=dtem_window_size,
                tome_window_size=tome_window_size,
            ).to(device)
            # 一些常用 DTEM 超参（与 train.py 对齐）
            model.local.vit._tome_info["k2"] = 4
            model.local.vit._tome_info["tau1"] = 1.0
            model.local.vit._tome_info["tau2"] = 0.1
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model.eval()

        # warmup
        for _ in range(max(0, warmup)):
            if mode == 'fwd':
                with torch.no_grad():
                    _ = model(x)
            else:
                out = model(x)
                logits = out[0] if isinstance(out, (tuple, list)) else (out['logits'] if isinstance(out, dict) and 'logits' in out else out)
                if not torch.is_tensor(logits):
                    raise TypeError(f"Model output type unsupported for backward: {type(out)}")
                loss = logits.sum()
                loss.backward()
                model.zero_grad(set_to_none=True)
            if use_cuda:
                torch.cuda.synchronize()

        # timed (CUDA events preferred on GPU)
        if use_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            for _ in range(max(1, iters)):
                if mode == 'fwd':
                    with torch.no_grad():
                        _ = model(x)
                else:
                    out = model(x)
                    logits = out[0] if isinstance(out, (tuple, list)) else (out['logits'] if isinstance(out, dict) and 'logits' in out else out)
                    if not torch.is_tensor(logits):
                        raise TypeError(f"Model output type unsupported for backward: {type(out)}")
                    loss = logits.sum()
                    loss.backward()
                    model.zero_grad(set_to_none=True)
            ender.record()
            torch.cuda.synchronize()
            total_ms = starter.elapsed_time(ender)
        else:
            t0 = time.perf_counter()
            for _ in range(max(1, iters)):
                if mode == 'fwd':
                    with torch.no_grad():
                        _ = model(x)
                else:
                    out = model(x)
                    logits = out[0] if isinstance(out, (tuple, list)) else (out['logits'] if isinstance(out, dict) and 'logits' in out else out)
                    if not torch.is_tensor(logits):
                        raise TypeError(f"Model output type unsupported for backward: {type(out)}")
                    loss = logits.sum()
                    loss.backward()
                    model.zero_grad(set_to_none=True)
            t1 = time.perf_counter()
            total_ms = (t1 - t0) * 1000.0

        total_imgs = batch_size * max(1, iters)
        total_time_s = max(1e-9, total_ms / 1000.0)
        ips = total_imgs / total_time_s
        ms_per_iter = total_ms / max(1, iters)
        return ips, ms_per_iter
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            if use_cuda:
                torch.cuda.empty_cache()
            if suppress_oom:
                return float('nan'), float('inf')
            # 不抑制时直接抛出，让外层决定是否继续
            raise
        raise


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark throughput/runtime for FlashAttentionModel and HybridToMeModel")
    parser.add_argument('--models', type=str, default='both', choices=['flash', 'hybrid', 'both'])
    parser.add_argument('--img_sizes', type=int, nargs='+', default=[224])
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--local_depth', type=int, default=4)
    parser.add_argument('--latent_depth', type=int, default=12)
    parser.add_argument('--dtem_window_size', type=int, default=8)
    parser.add_argument('--tome_window_size', type=int, default=None)
    parser.add_argument('--merge_local', type=int, default=8)
    parser.add_argument('--merge_latent', type=int, default=4)
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[32])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--mode', type=str, default='fwd', choices=['fwd', 'fwd_bwd'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='/yuchang/yk/OpenToMe/evaluations/throughputs/results')
    parser.add_argument('--output_file', type=str, default='throughput_models.csv')
    parser.add_argument('--suppress_oom', action='store_true', help='捕获 OOM 并记录 NaN/Inf，而不是抛出异常（默认不抑制）。')
    parser.add_argument('--continue_on_error', action='store_true', help='遇到异常时打印堆栈并继续下一个配置。')
    return parser.parse_args()


def main():
    args = get_args()

    torch.backends.cudnn.benchmark = True

    model_names = []
    if args.models in ('flash', 'both'):
        model_names.append('flash')
    if args.models in ('hybrid', 'both'):
        model_names.append('hybrid')

    rows = []
    for name in model_names:
        for img_size in args.img_sizes:
            for batch_size in args.batch_sizes:
                try:
                    ips, ms = bench_model(
                        model_name=name,
                        img_size=img_size,
                        patch_size=args.patch_size,
                        embed_dim=args.embed_dim,
                        num_heads=args.num_heads,
                        mlp_ratio=args.mlp_ratio,
                        local_depth=args.local_depth,
                        latent_depth=args.latent_depth,
                        dtem_window_size=args.dtem_window_size,
                        tome_window_size=args.tome_window_size,
                        total_merge_local=args.merge_local,
                        total_merge_latent=args.merge_latent,
                        batch_size=batch_size,
                        warmup=args.warmup,
                        iters=args.iters,
                        device=args.device,
                        mode=args.mode,
                        suppress_oom=args.suppress_oom,
                    )
                    print(f"{name} | img {img_size} | bs {batch_size} | {args.mode}: {ips:.2f} img/s, {ms:.2f} ms/iter")
                    rows.append({
                        'model': name,
                        'mode': args.mode,
                        'img_size': img_size,
                        'batch_size': batch_size,
                        'patch_size': args.patch_size,
                        'embed_dim': args.embed_dim,
                        'num_heads': args.num_heads,
                        'mlp_ratio': args.mlp_ratio,
                        'local_depth': args.local_depth,
                        'latent_depth': args.latent_depth,
                        'dtem_window_size': args.dtem_window_size,
                        'tome_window_size': args.tome_window_size,
                        'merge_local': args.merge_local,
                        'merge_latent': args.merge_latent,
                        'ips': ips,
                        'ms_per_iter': ms,
                        'error': ''
                    })
                except Exception as e:
                    # 打印完整堆栈，不隐藏错误
                    traceback.print_exc()
                    msg = f"{type(e).__name__}: {e}"
                    print(f"ERROR {name} | img {img_size} | bs {batch_size} | {args.mode}: {msg}")
                    if args.continue_on_error:
                        rows.append({
                            'model': name,
                            'mode': args.mode,
                            'img_size': img_size,
                            'batch_size': batch_size,
                            'patch_size': args.patch_size,
                            'embed_dim': args.embed_dim,
                            'num_heads': args.num_heads,
                            'mlp_ratio': args.mlp_ratio,
                            'local_depth': args.local_depth,
                            'latent_depth': args.latent_depth,
                            'dtem_window_size': args.dtem_window_size,
                            'tome_window_size': args.tome_window_size,
                            'merge_local': args.merge_local,
                            'merge_latent': args.merge_latent,
                            'ips': float('nan'),
                            'ms_per_iter': float('inf'),
                            'error': msg
                        })
                        continue
                    else:
                        # 不继续时直接抛出
                        raise

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved results to: {output_path}")


if __name__ == '__main__':
    main()


# python /yuchang/yk/OpenToMe/evaluations/throughputs/benchmark_throughput_models.py --models both --img_sizes 1000 2000 4000 --batch_sizes 16 32 64 --iters 100 --warmup 20 --mode fwd