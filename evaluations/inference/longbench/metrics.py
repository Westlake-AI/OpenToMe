import re
import string
from collections import Counter
from difflib import SequenceMatcher


CN_PUNCTUATION = (
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
    "｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
)


def normalize_answer(text):
    text = text.lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalize_zh_answer(text):
    punctuation = set(string.punctuation + CN_PUNCTUATION)
    return "".join(char for char in text.lower() if char not in punctuation and not char.isspace())


def _zh_tokens(text):
    try:
        import jieba
    except ImportError:
        return list(normalize_zh_answer(text))
    return [normalize_zh_answer(token) for token in jieba.cut(text, cut_all=False) if normalize_zh_answer(token)]


def f1_score(prediction, ground_truth):
    if not prediction or not ground_truth:
        return float(prediction == ground_truth)
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if not num_same:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    return f1_score(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    return f1_score(_zh_tokens(prediction), _zh_tokens(ground_truth))


def _lcs_length(first, second):
    previous = [0] * (len(second) + 1)
    for left in first:
        current = [0]
        for index, right in enumerate(second, start=1):
            if left == right:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l(prediction_tokens, ground_truth_tokens):
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    overlap = _lcs_length(prediction_tokens, ground_truth_tokens)
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 0.0 if not overlap else 2 * precision * recall / (precision + recall)


def rouge_score(prediction, ground_truth, **kwargs):
    return _rouge_l(prediction.split(), ground_truth.split())


def rouge_zh_score(prediction, ground_truth, **kwargs):
    return _rouge_l(_zh_tokens(prediction), _zh_tokens(ground_truth))


def classification_score(prediction, ground_truth, all_classes=None, **kwargs):
    matches = [name for name in (all_classes or []) if name in prediction]
    matches = [name for name in matches if not (name in ground_truth and name != ground_truth)]
    return 1.0 / len(matches) if ground_truth in matches else 0.0


def retrieval_score(prediction, ground_truth, **kwargs):
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    numbers = re.findall(r"\d+", prediction)
    return 0.0 if not numbers else sum(number == matches[0] for number in numbers) / len(numbers)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    matches = re.findall(r"段落\s*(\d+)", ground_truth)
    if not matches:
        return 0.0
    numbers = re.findall(r"\d+", prediction)
    return 0.0 if not numbers else sum(number == matches[0] for number in numbers) / len(numbers)


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    return 0.0 if not numbers else sum(number == str(ground_truth) for number in numbers) / len(numbers)


def code_sim_score(prediction, ground_truth, **kwargs):
    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    return SequenceMatcher(None, candidate, ground_truth).ratio()


DATASET_TO_METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


FIRST_LINE_DATASETS = {"trec", "triviaqa", "samsum", "lsht"}


def score_record(dataset, record):
    prediction = record["pred"]
    if dataset in FIRST_LINE_DATASETS:
        prediction = prediction.lstrip("\n").split("\n")[0]
    answers = record.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    metric = DATASET_TO_METRIC[dataset]
    return max(
        (metric(prediction, answer, all_classes=record.get("all_classes", [])) for answer in answers),
        default=0.0,
    )


def score_records(dataset, records, longbench_e=False):
    if dataset not in DATASET_TO_METRIC:
        raise ValueError(f"No LongBench metric registered for {dataset!r}")
    if not records:
        raise ValueError("Cannot score an empty prediction set")
    if not longbench_e:
        return round(100 * sum(score_record(dataset, record) for record in records) / len(records), 2)

    buckets = {"0-4k": [], "4-8k": [], "8k+": []}
    for record in records:
        length = int(record.get("length", 0))
        bucket = "0-4k" if length < 4000 else "4-8k" if length < 8000 else "8k+"
        buckets[bucket].append(score_record(dataset, record))
    return {
        name: round(100 * sum(values) / len(values), 2) if values else None
        for name, values in buckets.items()
    }


__all__ = ["DATASET_TO_METRIC", "score_record", "score_records"]

