import json
from pathlib import Path


LONG_BENCH_DATASETS = (
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report",
    "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum",
    "lsht", "passage_count", "passage_retrieval_en",
    "passage_retrieval_zh", "lcc", "repobench-p",
)

LONG_BENCH_E_DATASETS = (
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
    "multi_news", "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
)

REPOSITORY_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "LongBench"


def _read_local(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def resolve_local_path(local_data: Path, dataset: str, longbench_e: bool) -> Path:
    if local_data.is_file():
        return local_data
    name = f"{dataset}_e" if longbench_e else dataset
    for suffix in (".jsonl", ".json"):
        candidate = local_data / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {name}.jsonl or {name}.json under {local_data}")


def _repository_local_path(dataset: str, longbench_e: bool):
    try:
        return resolve_local_path(REPOSITORY_DATA_DIR, dataset, longbench_e)
    except FileNotFoundError:
        return None


def _load_hub_file(dataset_path: str, subset: str):
    """Download a raw JSONL when recent datasets versions ignore dataset configs."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    for filename in (f"{subset}.jsonl", f"data/{subset}.jsonl"):
        try:
            path = hf_hub_download(
                repo_id=dataset_path, filename=filename, repo_type="dataset"
            )
        except Exception:
            continue
        return _read_local(Path(path))
    return None


def _load_default_config(load_dataset, dataset_path: str, subset: str):
    """Load and filter a converted single-config LongBench dataset."""
    errors = []
    for split in ("test", "train"):
        try:
            records = load_dataset(dataset_path, "default", split=split)
        except (ValueError, KeyError) as exc:
            errors.append(exc)
            continue
        columns = getattr(records, "column_names", ())
        if "dataset" not in columns:
            raise ValueError(
                f"{dataset_path!r} default config has no 'dataset' column; "
                "pass --local-data with the LongBench JSONL directory"
            )
        return records.filter(lambda record: record.get("dataset") == subset)
    raise errors[-1]


def load_longbench_records(
    dataset: str,
    longbench_e: bool = False,
    local_data: Path = None,
    dataset_path: str = "THUDM/LongBench",
):
    if local_data is not None:
        return _read_local(resolve_local_path(Path(local_data), dataset, longbench_e))

    repository_path = _repository_local_path(dataset, longbench_e)
    if repository_path is not None:
        return _read_local(repository_path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "LongBench data was not found under data/LongBench and online loading "
            "requires `datasets`; install it or pass --local-data"
        ) from exc

    subset = f"{dataset}_e" if longbench_e else dataset
    try:
        return load_dataset(dataset_path, subset, split="test")
    except ValueError as exc:
        if "BuilderConfig" not in str(exc) and "config" not in str(exc).lower():
            raise
        raw_records = _load_hub_file(dataset_path, subset)
        if raw_records is not None:
            return raw_records
        return _load_default_config(load_dataset, dataset_path, subset)


def normalize_dataset_args(values, longbench_e=False):
    available = LONG_BENCH_E_DATASETS if longbench_e else LONG_BENCH_DATASETS
    if not values or values == ["all"]:
        return list(available)
    datasets = []
    for value in values:
        datasets.extend(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(datasets) - set(available))
    if unknown:
        raise ValueError(f"Unsupported LongBench datasets: {', '.join(unknown)}")
    return list(dict.fromkeys(datasets))
