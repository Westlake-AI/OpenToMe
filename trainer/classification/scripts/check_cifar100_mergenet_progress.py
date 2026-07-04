#!/usr/bin/env python3
"""Read-only progress checker for the CIFAR-100 MergeNet-vs-DeiT runs.

Scans work_dirs/classification run directories, reads summary.csv, and prints:
  - best / last eval top1 (and best restricted to full-compression epochs when a
    lambda curriculum is active, via the eval_top1_full_compression column);
  - ETA estimated from the summary.csv mtime cadence;
  - whether the run already beats the two reference numbers:
      75.12 (single MergeNet FT200 non-distill) and 80.67 (DeiT FT200).

Never modifies or kills anything.

Usage:
  python check_cifar100_mergenet_progress.py                      # default glob
  python check_cifar100_mergenet_progress.py --pattern 'cifar100_mn_ft200_*'
  python check_cifar100_mergenet_progress.py --run <run_dir> [--watch 60]
"""

import argparse
import csv
import glob
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

MN_BASELINE = 75.12   # single MergeNet-B FT200, no distill (2026-06-29)
DEIT_BASELINE = 80.67  # DeiT FT200 from the same checkpoint (2026-06-29)


def _f(row, key, default=None):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def read_total_epochs(run_dir: Path) -> int:
    args_yaml = run_dir / "args.yaml"
    if args_yaml.is_file():
        for line in args_yaml.read_text(errors="ignore").splitlines():
            if line.startswith("epochs:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    break
    return 200


def analyze_run(run_dir: Path):
    summary = run_dir / "summary.csv"
    if not summary.is_file():
        return None
    with open(summary, newline="") as f:
        rows = list(csv.DictReader(f))
    # Tolerate legacy files with a repeated header row after resume.
    rows = [r for r in rows if str(r.get("epoch", "")).strip().lstrip("-").isdigit()]
    if not rows:
        return None

    total_epochs = read_total_epochs(run_dir)
    done_epochs = len(rows)
    last = rows[-1]

    best_row = max(rows, key=lambda r: _f(r, "eval_top1", -1.0))
    best_top1 = _f(best_row, "eval_top1")
    best_epoch = int(_f(best_row, "epoch", -1))

    fair_best_top1, fair_best_epoch = None, None
    if any("eval_top1_full_compression" in r for r in rows):
        fair_rows = [r for r in rows if _f(r, "eval_top1_full_compression", 0.0) > 0.0]
        if fair_rows:
            fb = max(fair_rows, key=lambda r: _f(r, "eval_top1_full_compression", -1.0))
            fair_best_top1 = _f(fb, "eval_top1_full_compression")
            fair_best_epoch = int(_f(fb, "epoch", -1))

    mtime = summary.stat().st_mtime
    age_s = time.time() - mtime
    # cadence: assume uniform epoch time measured from ctime of the run dir.
    try:
        start = run_dir.stat().st_mtime if done_epochs <= 1 else None
    except OSError:
        start = None
    per_epoch_s = None
    if done_epochs >= 2:
        # infer per-epoch seconds from throughput column when available
        thr = _f(last, "train_throughput")
        if thr and thr > 0:
            per_epoch_s = 50000.0 / thr + 10.0  # CIFAR-100 train split + eval overhead

    remaining = max(total_epochs - done_epochs, 0)
    eta = None
    if per_epoch_s and remaining:
        eta = timedelta(seconds=int(per_epoch_s * remaining))

    active = age_s < 30 * 60

    return {
        "run": run_dir.name,
        "epochs": f"{done_epochs}/{total_epochs}",
        "best_top1": best_top1,
        "best_epoch": best_epoch,
        "fair_best_top1": fair_best_top1,
        "fair_best_epoch": fair_best_epoch,
        "last_top1": _f(last, "eval_top1"),
        "last_lambda": _f(last, "train_effective_lambda"),
        "last_retained": _f(last, "train_retained_tokens"),
        "active": active,
        "updated_min_ago": age_s / 60.0,
        "eta": eta,
    }


def verdict(top1):
    if top1 is None:
        return "n/a"
    parts = []
    parts.append(f"{'>' if top1 > MN_BASELINE else '<='} MN75.12")
    parts.append(f"{'>' if top1 > DEIT_BASELINE else '<='} DeiT80.67")
    return ", ".join(parts)


def report(root: Path, pattern: str, explicit_runs):
    run_dirs = [Path(r) for r in explicit_runs] if explicit_runs else sorted(
        Path(p) for p in glob.glob(str(root / pattern)) if os.path.isdir(p))
    if not run_dirs:
        print(f"[progress] no runs matched {root}/{pattern}")
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[progress] {now}  (baselines: single-MN {MN_BASELINE}, DeiT {DEIT_BASELINE})")
    for run_dir in run_dirs:
        info = analyze_run(run_dir)
        if info is None:
            print(f"  - {run_dir.name}: no summary.csv yet")
            continue
        state = "RUNNING" if info["active"] else "idle"
        fair = ""
        if info["fair_best_top1"] is not None:
            fair = (f"  fair_best(full-compress)={info['fair_best_top1']:.2f}"
                    f"@{info['fair_best_epoch']} [{verdict(info['fair_best_top1'])}]")
        elif info["fair_best_top1"] is None and info["last_lambda"] is not None:
            fair = "  fair_best=none-yet (curriculum still ramping)"
        lam = f"  lambda={info['last_lambda']:.2f}" if info["last_lambda"] is not None else ""
        ret = f" retained={int(info['last_retained'])}" if info["last_retained"] else ""
        eta = f"  ETA~{info['eta']}" if info["eta"] else ""
        print(
            f"  - {info['run']}\n"
            f"      [{state}] epoch {info['epochs']}  updated {info['updated_min_ago']:.0f}m ago{eta}\n"
            f"      best={info['best_top1']:.2f}@{info['best_epoch']} [{verdict(info['best_top1'])}]"
            f"  last={info['last_top1']:.2f}{lam}{ret}{fair}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = Path(__file__).resolve().parents[3] / "work_dirs" / "classification"
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--pattern", default="cifar100_*ft200*")
    parser.add_argument("--run", action="append", default=[], help="explicit run dir(s); overrides --pattern")
    parser.add_argument("--watch", type=int, default=0, help="repeat every N seconds (0 = once)")
    args = parser.parse_args()

    while True:
        report(args.root, args.pattern, args.run)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
