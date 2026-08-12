import json
import tempfile
import unittest
from pathlib import Path

import torch

from evaluations.inference.longbench.data import load_longbench_records, normalize_dataset_args
from evaluations.inference.longbench.metrics import (
    classification_score,
    qa_f1_score,
    retrieval_score,
    rouge_score,
    score_records,
)
from evaluations.inference.longbench.predict import middle_truncate
from opentome.compress import H2OPolicy, PyramidKVPolicy, SnapKVPolicy, StreamingKVPolicy


class LongBenchEvaluationTest(unittest.TestCase):
    def test_metrics_and_dataset_scoring(self):
        self.assertEqual(qa_f1_score("The red fox", "red fox"), 1.0)
        self.assertEqual(rouge_score("a b c", "a b c"), 1.0)
        self.assertEqual(retrieval_score("Paragraph 3", "Paragraph 3"), 1.0)
        self.assertEqual(
            classification_score("HUM", "HUM", all_classes=["HUM", "LOC"]), 1.0
        )
        records = [
            {"pred": "answer", "answers": ["answer"], "length": 2000},
            {"pred": "wrong", "answers": ["answer"], "length": 6000},
            {"pred": "answer", "answers": ["answer"], "length": 9000},
        ]
        self.assertEqual(score_records("qasper", records), 66.67)
        self.assertEqual(
            score_records("qasper", records, longbench_e=True),
            {"0-4k": 100.0, "4-8k": 0.0, "8k+": 100.0},
        )

    def test_local_jsonl_loading_and_dataset_args(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qasper.jsonl"
            path.write_text(json.dumps({"context": "c", "input": "q"}) + "\n", encoding="utf-8")
            records = load_longbench_records("qasper", local_data=Path(directory))
        self.assertEqual(records[0]["context"], "c")
        self.assertEqual(normalize_dataset_args(["qasper,hotpotqa"]), ["qasper", "hotpotqa"])

    def test_middle_truncate(self):
        tokens = torch.arange(10).view(1, -1)
        self.assertEqual(middle_truncate(tokens, 6).tolist(), [[0, 1, 2, 7, 8, 9]])

    def test_policies_are_under_methods_package(self):
        self.assertEqual(StreamingKVPolicy.__module__, "opentome.compress.methods.streamingkv")
        self.assertEqual(H2OPolicy.__module__, "opentome.compress.methods.h2o")
        self.assertEqual(SnapKVPolicy.__module__, "opentome.compress.methods.snapkv")
        self.assertEqual(PyramidKVPolicy.__module__, "opentome.compress.methods.pyramidkv")


if __name__ == "__main__":
    unittest.main()
