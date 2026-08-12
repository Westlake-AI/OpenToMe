import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder-path",
        "--folder_path",
        type=Path,
        default=REPO_ROOT / "work_dirs" / "needle" / "results" / "LLaMA-2-7B-32K",
        help="Directory containing Needle JSON results",
    )
    parser.add_argument("--model-name", "--model_name", default="LLaMA-2-7B-32K")
    parser.add_argument("--pretrained-len", "--pretrained_len", type=int, default=32000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "work_dirs" / "needle" / "visualizations",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.folder_path.name or args.model_name
    rows = []
    for path in sorted(args.folder_path.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        response = str(data.get("model_response", "")).lower()
        expected = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
        score = len(set(response.split()).intersection(expected)) / len(expected)
        rows.append(
            {
                "Document Depth": data.get("depth_percent"),
                "Context Length": data.get("context_length"),
                "Score": score,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No Needle JSON results found under {args.folder_path}")

    frame = pd.DataFrame(rows)
    lengths = sorted(frame["Context Length"].dropna().unique())
    pretrained_boundary = sum(length <= args.pretrained_len for length in lengths)
    print(f"Overall score {frame['Score'].mean():.3f}")

    pivot = pd.pivot_table(
        frame,
        values="Score",
        index=["Document Depth", "Context Length"],
        aggfunc="mean",
    ).reset_index()
    pivot = pivot.pivot(
        index="Document Depth", columns="Context Length", values="Score"
    )
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
    )
    plt.figure(figsize=(17.5, 8))
    sns.heatmap(
        pivot,
        vmin=0,
        vmax=1,
        cmap=cmap,
        cbar_kws={"label": "Score"},
        linewidths=0.5,
        linecolor="grey",
    )
    plt.title(f"NIAH {model_name}\nOverall Score: {frame['Score'].mean():.3f}")
    plt.xlabel("Token Limit")
    plt.ylabel("Depth Percent")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if pretrained_boundary < len(lengths):
        plt.axvline(
            x=pretrained_boundary,
            color="white",
            linestyle="--",
            linewidth=4,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{model_name}.png"
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
