#!/usr/bin/env python3
"""Plot accuracy / throughput / memory / LR curves for the scratch-200e campaign.

Modeled on /liziqing/yukai/plot_summary_metrics.py, but:
  - auto-discovers campaign runs (c100_scratch*) under the work dir, so newly
    launched experiments show up without editing this file;
  - tolerates duplicated CSV headers written on resume;
  - additionally plots train_throughput and the MergeNet-specific columns
    (effective lambda / retained tokens) when present.

Usage:
  python plot_scratch200_results.py                          # auto-discover
  python plot_scratch200_results.py --model path/to/summary.csv:label ...
  python plot_scratch200_results.py --glob 'c100_scratch200_*' --output-dir plots
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

DEFAULT_WORK_DIR = Path(__file__).resolve().parents[3] / "work_dirs" / "classification"


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop duplicated header rows that timm's update_summary can write on resume.
    df = df[pd.to_numeric(df["epoch"], errors="coerce").notna()].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("epoch")


def discover_models(work_dir: Path, pattern: str):
    models = []
    for exp_dir in sorted(work_dir.glob(pattern)):
        csv = exp_dir / "summary.csv"
        if csv.exists():
            models.append((csv, exp_dir.name))
    return models


def series_for(models, column: str):
    out = []
    for csv, label in models:
        df = load_summary(Path(csv))
        if column not in df.columns:
            continue
        part = df[["epoch", column]].dropna()
        if part.empty:
            continue
        out.append((label, part["epoch"].to_numpy(), part[column].to_numpy()))
    return out


def shorten(label: str) -> str:
    return label.replace("c100_scratch200_", "").replace("_p8_b200", "")


def plot_metric(series, ylabel, title, out_path: Path, annotate_best=False,
                baseline=None, baseline_label="baseline"):
    if not series:
        return None
    plt.figure(figsize=(8.5, 5), dpi=150)
    if baseline is not None:
        plt.axhline(baseline, color="gray", linestyle=":", linewidth=1.2,
                    label=f"{baseline_label} ({baseline:.2f})")
    for label, epochs, values in series:
        (line,) = plt.plot(epochs, values, linewidth=1.7, label=shorten(label))
        if annotate_best and len(values):
            best_i = values.argmax()
            plt.scatter([epochs[best_i]], [values[best_i]], s=22,
                        color=line.get_color(), zorder=5)
            plt.annotate(f"{values[best_i]:.2f}", (epochs[best_i], values[best_i]),
                         textcoords="offset points", xytext=(4, 4),
                         fontsize=7, color=line.get_color())
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    if "learning rate" in ylabel.lower():
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    plt.legend(fontsize="small")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path


def print_table(models):
    rows = []
    for csv, label in models:
        df = load_summary(Path(csv))
        if df.empty or "eval_top1" not in df.columns:
            continue
        best_i = df["eval_top1"].idxmax()
        row = {
            "experiment": shorten(label),
            "done_ep": int(df["epoch"].max()),
            "best_top1": round(float(df.loc[best_i, "eval_top1"]), 2),
            "best_ep": int(df.loc[best_i, "epoch"]),
            "last_top1": round(float(df["eval_top1"].iloc[-1]), 2),
        }
        for col, name in (("eval_throughput", "eval_imgs_s"),
                          ("train_throughput", "train_imgs_s"),
                          ("eval_mem_allocated_mb", "eval_mem_mb"),
                          ("train_mem_allocated_mb", "train_mem_mb")):
            if col in df.columns and df[col].notna().any():
                row[name] = round(float(df[col].dropna().iloc[-1]), 1)
        if "eval_top1_full_compression" in df.columns and df["eval_top1_full_compression"].notna().any():
            fc = df.dropna(subset=["eval_top1_full_compression"])
            row["fair_best_top1"] = round(float(fc["eval_top1_full_compression"].max()), 2)
        rows.append(row)
    if rows:
        table = pd.DataFrame(rows)
        print(table.to_string(index=False))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR),
                        help="Directory containing experiment folders.")
    parser.add_argument("--glob", default="c100_scratch*",
                        help="Glob pattern for auto-discovering experiments.")
    parser.add_argument("--baseline", type=float, default=None,
                        help="Draw horizontal baseline line on accuracy plot (e.g. 67.18).")
    parser.add_argument("--baseline-label", default="deit baseline",
                        help="Legend label for --baseline.")
    parser.add_argument("--model", action="append", default=[],
                        help="Extra/override runs: /path/summary.csv:label (repeatable).")
    parser.add_argument("--output-dir",
                        default=str(DEFAULT_WORK_DIR / "campaign_plots"),
                        help="Directory to save plots.")
    args = parser.parse_args()

    models = discover_models(Path(args.work_dir), args.glob)
    for item in args.model:
        path, _, label = item.partition(":")
        models.append((Path(path), label or Path(path).parent.name))
    if not models:
        raise SystemExit(f"no summary.csv found under {args.work_dir}/{args.glob}")

    out = Path(args.output_dir)
    print(f"[plot] {len(models)} runs:")
    print_table(models)

    written = []
    for column, ylabel, title, fname, annotate in [
        ("eval_top1", "top1 accuracy", "Eval Top-1 vs Epoch", "accuracy_top1.png", True),
        ("eval_throughput", "eval throughput (img/s)", "Eval Throughput vs Epoch", "eval_throughput.png", False),
        ("train_throughput", "train throughput (img/s)", "Train Throughput vs Epoch", "train_throughput.png", False),
        ("eval_mem_allocated_mb", "eval memory (MB)", "Eval Memory vs Epoch", "eval_memory_mb.png", False),
        ("train_mem_allocated_mb", "train memory (MB)", "Train Memory vs Epoch", "train_memory_mb.png", False),
        ("train_lr", "learning rate", "Learning Rate vs Epoch", "learning_rate.png", False),
        ("train_effective_lambda", "effective lambda", "Compression Curriculum", "effective_lambda.png", False),
        ("train_retained_tokens", "retained tokens", "Retained Tokens vs Epoch", "retained_tokens.png", False),
        ("train_loss", "train loss", "Train Loss vs Epoch", "train_loss.png", False),
    ]:
        path = plot_metric(series_for(models, column), ylabel, title, out / fname,
                           annotate_best=annotate,
                           baseline=args.baseline if column == "eval_top1" else None,
                           baseline_label=args.baseline_label)
        if path:
            written.append(path)

    print("图像已生成：")
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
