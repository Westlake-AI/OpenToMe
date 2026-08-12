import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluations.inference.longbench.metrics import score_records


def parse_args():
    parser = argparse.ArgumentParser(description="Score OpenToMe LongBench JSONL predictions")
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--dataset", action="append", help="Required for a file; optional filter for a directory")
    parser.add_argument("--longbench-e", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_prediction_files(path, datasets=None):
    if path.is_file():
        if datasets and len(datasets) != 1:
            raise ValueError("Exactly one --dataset is required when scoring one file")
        return [(datasets[0] if datasets else path.stem, path)]
    selected = set(datasets or [])
    files = []
    for candidate in sorted(path.glob("*.jsonl")):
        if not selected or candidate.stem in selected:
            files.append((candidate.stem, candidate))
    if not files:
        raise FileNotFoundError(f"No matching JSONL prediction files under {path}")
    return files


def evaluate_path(path, datasets=None, longbench_e=False):
    results = {}
    for dataset, prediction_file in resolve_prediction_files(Path(path), datasets):
        results[dataset] = score_records(dataset, read_jsonl(prediction_file), longbench_e)
    return results


def main():
    args = parse_args()
    datasets = []
    for value in args.dataset or []:
        datasets.extend(item.strip() for item in value.split(",") if item.strip())
    results = evaluate_path(args.prediction_path, datasets or None, args.longbench_e)
    output = args.output or (
        args.prediction_path / "result.json" if args.prediction_path.is_dir()
        else args.prediction_path.with_name(f"{args.prediction_path.stem}_result.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Scores written to {output}")


if __name__ == "__main__":
    main()
