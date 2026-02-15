"""
绘制 timm 训练 summary.csv 的收敛曲线。

用法示例：
python tools/plot_summary.py \
  --csv /liziqing/yuhao/yukai/OpenToMe/work_dirs/classification/cifar100_mergenet_small_260115/summary.csv \
  --out /liziqing/yuhao/yukai/OpenToMe/work_dirs/classification/cifar100_mergenet_small_260115/summary.png

输出：
- 默认生成 PNG，包含 train/val loss、train/val top1（若存在）曲线。
- 自动跳过缺失的列。
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot convergence curves from summary.csv")
    p.add_argument("--csv", required=True, help="Path to summary.csv")
    p.add_argument("--out", default=None, help="Output image path (png). If None, auto set beside csv.")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    # 可能的列名
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    top1_cols = [c for c in df.columns if "top1" in c.lower()]
    top5_cols = [c for c in df.columns if "top5" in c.lower()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=args.dpi)

    # Loss
    ax = axes[0]
    plotted = False
    for c in loss_cols:
        ax.plot(df.index, df[c], label=c)
        plotted = True
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No loss columns", ha="center", va="center", transform=ax.transAxes)

    # Top-1 / Top-5
    ax = axes[1]
    plotted = False
    for c in top1_cols:
        ax.plot(df.index, df[c], label=c)
        plotted = True
    for c in top5_cols:
        ax.plot(df.index, df[c], label=c, linestyle="--")
        plotted = True
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No accuracy columns", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    out_path = args.out
    if out_path is None:
        base = os.path.splitext(args.csv)[0]
        out_path = base + ".png"
    plt.savefig(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
