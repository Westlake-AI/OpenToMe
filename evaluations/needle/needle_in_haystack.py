"""Needle-in-a-haystack evaluation with optional KV cache compression.

Adapted from https://github.com/gkamradt/LLMTest_NeedleInAHaystack.
"""

import argparse
import re
import glob
import json
import os
import time
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from evaluations.inference.kv_utils import (
    METHOD_CHOICES,
    build_cache,
    cache_stats,
    resolve_device,
    resolve_dtype,
    run_metadata,
    write_json,
)
from opentome.compress import CompressedDynamicCache
from opentome.models.kv_compression import patch_model_for_kv_compression


SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True) if rouge_scorer else None


def needle_rouge1_f1(reference, prediction):
    if SCORER is not None:
        return SCORER.score(reference, prediction)["rouge1"].fmeasure
    reference_tokens = re.findall(r"\w+", reference.lower())
    prediction_tokens = re.findall(r"\w+", prediction.lower())
    if not reference_tokens or not prediction_tokens:
        return 0.0
    overlap = sum((Counter(reference_tokens) & Counter(prediction_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _load_backbone_registration():
    backbone = os.environ.get("BACKBONE", "None")
    if "gated_deltanet" in backbone:
        import fla.models.gated_deltanet  # noqa: F401
    elif "delta_net" in backbone:
        import fla.models.delta_net  # noqa: F401
    elif "gla" in backbone:
        import fla.models.gla  # noqa: F401
    elif "transformer" in backbone:
        import fla.models.transformer  # noqa: F401


class LLMNeedleHaystackTester:
    def __init__(
        self,
        args,
        needle="\n\nRemember, the best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n",
        haystack_dir="PaulGrahamEssays",
        retrieval_question="what is the best thing to do in San Francisco?\n\nAnswer: The best thing to do in San Francisco is",
        results_version=1,
        context_lengths_min=1000,
        context_lengths_max=1048000,
        context_lengths_num_intervals=40,
        context_lengths=None,
        document_depth_percent_min=0,
        document_depth_percent_max=100,
        document_depth_percent_intervals=10,
        document_depth_percents=None,
        document_depth_percent_interval_type="linear",
        model_name="",
        model_name_suffix=None,
        tokenizer_path=None,
        save_results=True,
        save_contexts=True,
        final_context_length_buffer=200,
        seconds_to_sleep_between_completions=None,
        print_ongoing_status=True,
        simulation_length=50,
    ):
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided")
        if simulation_length < 0:
            raise ValueError("simulation_length must be non-negative")

        self.args = args
        self.work_dir = Path(args.work_dir)
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.save_results = save_results
        self.save_contexts = save_contexts
        self.final_context_length_buffer = final_context_length_buffer
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []
        self.simulation_length = simulation_length
        self.model_name = model_name
        self.method = args.method

        self.model_version = Path(model_name).name
        if model_name_suffix:
            self.model_version += f"_{model_name_suffix}"
        if self.method != "none":
            self.model_version += f"_{self.method}_{args.max_capacity_prompt}"

        self.context_lengths = self._context_lengths(
            context_lengths,
            context_lengths_min,
            context_lengths_max,
            context_lengths_num_intervals,
        )
        self.document_depth_percents = self._document_depths(
            document_depth_percents,
            document_depth_percent_min,
            document_depth_percent_max,
            document_depth_percent_intervals,
            document_depth_percent_interval_type,
        )

        tokenizer_path = tokenizer_path or model_name
        self.enc = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=args.use_fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
        if self.enc.pad_token_id is None:
            self.enc.pad_token_id = self.enc.eos_token_id if self.enc.eos_token_id is not None else 0
        try:
            generation_config = GenerationConfig.from_pretrained(model_name)
            eos_token_ids = generation_config.eos_token_id
        except OSError:
            eos_token_ids = self.enc.eos_token_id
        self.eos_token_ids = eos_token_ids if isinstance(eos_token_ids, list) else [eos_token_ids]
        self.eos_token_ids = [token for token in self.eos_token_ids if token is not None]

        device = resolve_device(args.device)
        dtype = resolve_dtype(args.dtype, device)
        model_kwargs = {
            "torch_dtype": dtype,
            "attn_implementation": "eager",
            "trust_remote_code": args.trust_remote_code,
        }
        if args.device_map:
            model_kwargs["device_map"] = args.device_map
            self.model_to_test = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            ).eval()
        else:
            self.model_to_test = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            ).to(device).eval()
        if self.method != "none":
            patch_model_for_kv_compression(self.model_to_test)
        self.device = next(self.model_to_test.parameters()).device
        self.model_to_test_description = model_name

        if self.save_results:
            write_json(self.work_dir / "results" / self.model_version / "metadata.json", run_metadata(args))

    @staticmethod
    def _context_lengths(values, minimum, maximum, intervals):
        if values is not None:
            return np.asarray(values, dtype=int)
        if minimum is None or maximum is None or intervals is None:
            raise ValueError("Context length range or explicit values are required")
        return np.round(np.linspace(minimum, maximum, num=intervals, endpoint=True)).astype(int)

    @staticmethod
    def _document_depths(values, minimum, maximum, intervals, interval_type):
        if values is not None:
            return values
        if minimum is None or maximum is None or intervals is None:
            raise ValueError("Document depth range or explicit values are required")
        points = np.linspace(minimum, maximum, intervals)
        if interval_type == "linear":
            return np.round(points).astype(int)
        if interval_type == "sigmoid":
            return [LLMNeedleHaystackTester.logistic(x) for x in points]
        raise ValueError("document_depth_percent_interval_type must be linear or sigmoid")

    @staticmethod
    def logistic(x, L=100, x0=50, k=0.1):
        if x == 0 or x == 100:
            return x
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def run_test(self, args):
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len:
                continue
            for depth_percent in self.document_depth_percents:
                self.evaluate_and_log(context_length, depth_percent)

    def generate_prompt(self, context):
        return (
            f"<|im_start|> This is a very long story book: <book> {context} </book>.\n\n"
            f"Question: Based on the content of the book, {self.retrieval_question}"
        )

    def _position_ids(self, cache, token_count):
        if isinstance(cache, CompressedDynamicCache):
            start = cache.get_logical_length()
        else:
            start = cache.get_seq_length()
        return torch.arange(start, start + token_count, device=self.device).unsqueeze(0)

    def _model_step(self, input_ids, cache):
        kwargs = {"input_ids": input_ids, "past_key_values": cache, "use_cache": True}
        if isinstance(cache, CompressedDynamicCache):
            kwargs["position_ids"] = self._position_ids(cache, input_ids.shape[-1])
        output = self.model_to_test(**kwargs)
        return output, output.past_key_values

    @torch.inference_mode()
    def _generate_response(self, prompt_input_ids):
        cache = build_cache(self.args, self.model_to_test) if self.method != "none" else None
        suffix_length = min(self.simulation_length, max(0, prompt_input_ids.shape[-1] - 1))
        if suffix_length:
            prefill_ids = prompt_input_ids[:, :-suffix_length]
            suffix_ids = prompt_input_ids[:, -suffix_length:]
        else:
            prefill_ids = prompt_input_ids
            suffix_ids = prompt_input_ids[:, :0]

        if self.args.prefilling_chunk_size is None:
            output, cache = self._model_step(prefill_ids, cache)
        else:
            output = None
            for offset in range(0, prefill_ids.shape[-1], self.args.prefilling_chunk_size):
                output, cache = self._model_step(
                    prefill_ids[:, offset : offset + self.args.prefilling_chunk_size], cache
                )
        for input_id in suffix_ids[0]:
            output, cache = self._model_step(input_id.view(1, 1), cache)
        if output is None:
            raise ValueError("Prompt produced no input tokens")

        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = [next_token.item()]
        for _ in range(self.args.max_new_tokens - 1):
            output, cache = self._model_step(next_token, cache)
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())
            if next_token.item() in self.eos_token_ids:
                break
        return self.enc.decode(generated, skip_special_tokens=True).strip(), cache

    def evaluate_and_log(self, context_length, depth_percent):
        if self.save_results and self.result_exists(context_length, depth_percent):
            print("result exists, skipping")
            return
        context = self.generate_context(context_length, depth_percent)
        prompt = self.generate_prompt(context)
        prompt_input_ids = self.enc(prompt, return_tensors="pt")["input_ids"].to(self.device)

        start = time.perf_counter()
        response, cache = self._generate_response(prompt_input_ids)
        elapsed = time.perf_counter() - start
        score = needle_rouge1_f1(self.needle, response) * 10
        results = {
            "model": self.model_to_test_description,
            "method": self.method,
            "context_length": int(context_length),
            "depth_percent": float(depth_percent),
            "version": self.results_version,
            "needle": self.needle,
            "model_response": response,
            "score": score,
            "test_duration_seconds": elapsed,
            "test_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
            **cache_stats(cache),
        }
        self.testing_results.append(results)
        if self.print_ongoing_status:
            print(
                f"Duration: {elapsed:.1f}s | Context: {context_length} | "
                f"Depth: {depth_percent}% | Score: {score}\nResponse: {response}\n"
            )
        self._save_case(context, context_length, depth_percent, results)
        if self.seconds_to_sleep_between_completions:
            time.sleep(self.seconds_to_sleep_between_completions)

    def _case_name(self, context_length, depth_percent):
        return (
            f'{self.model_version.replace(".", "_")}_len_{context_length}_'
            f"depth_{int(depth_percent * 100)}"
        )

    def _save_case(self, context, context_length, depth_percent, results):
        case_name = self._case_name(context_length, depth_percent)
        if self.save_contexts:
            context_dir = self.work_dir / "contexts" / self.model_version
            context_dir.mkdir(parents=True, exist_ok=True)
            results["file_name"] = case_name
            (context_dir / f"{case_name}_context.txt").write_text(context, encoding="utf-8")
        if self.save_results:
            result_dir = self.work_dir / "results" / self.model_version
            result_dir.mkdir(parents=True, exist_ok=True)
            write_json(result_dir / f"{case_name}_results.json", results)

    def result_exists(self, context_length, depth_percent):
        result_path = self.work_dir / "results" / self.model_version / (
            self._case_name(context_length, depth_percent) + "_results.json"
        )
        if not result_path.exists():
            return False
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return (
            result.get("context_length") == context_length
            and result.get("depth_percent") == depth_percent
            and result.get("version", 1) == self.results_version
            and result.get("model") == self.model_name
            and result.get("method", "none") == self.method
        )

    def generate_context(self, context_length, depth_percent):
        context = self.encode_and_trim(self.read_context_files(), context_length)
        return self.insert_needle(context, depth_percent, context_length)

    def encode_text_to_tokens(self, text):
        return self.enc.encode(text, add_special_tokens=False)

    def insert_needle(self, context, depth_percent, context_length):
        needle_tokens = self.encode_text_to_tokens(self.needle)
        context_tokens = self.encode_text_to_tokens(context)
        available = context_length - self.final_context_length_buffer - len(needle_tokens)
        if available < 0:
            raise ValueError("context_length is smaller than the prompt buffer and needle")
        context_tokens = context_tokens[:available]
        insertion = len(context_tokens) if depth_percent == 100 else int(
            len(context_tokens) * depth_percent / 100
        )
        return self.decode_tokens(
            context_tokens[:insertion] + needle_tokens + context_tokens[insertion:]
        )

    def get_context_length_in_tokens(self, context):
        return len(self.enc.encode(context, add_special_tokens=False))

    def read_context_files(self):
        files = sorted(glob.glob(f"{self.haystack_dir}/*.txt"))
        if not files:
            raise ValueError(f"No .txt files found under {self.haystack_dir}")
        texts = [Path(path).read_text(encoding="utf-8", errors="ignore") for path in files]
        block = "".join(texts)
        if not block:
            raise ValueError("Haystack files are empty")
        context = block
        target = max(self.context_lengths)
        while self.get_context_length_in_tokens(context) < target:
            context += block
        return context

    def decode_tokens(self, tokens, context_length=None):
        return self.enc.decode(tokens[:context_length], skip_special_tokens=True)

    def encode_and_trim(self, context, context_length):
        tokens = self.enc.encode(context, add_special_tokens=False)
        return self.decode_tokens(tokens, context_length) if len(tokens) > context_length else context

    def get_results(self):
        return self.testing_results

    def start_test(self, args):
        if self.print_ongoing_status:
            print(
                "Starting Needle In A Haystack Testing...\n"
                f"- Model: {self.model_name}\n- Method: {self.method}\n"
                f"- Context lengths: {min(self.context_lengths)}..{max(self.context_lengths)}\n"
                f"- Document depths: {min(self.document_depth_percents)}..{max(self.document_depth_percents)}%"
            )
        self.run_test(args)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--s-len", "--s_len", dest="s_len", type=int, required=True)
    parser.add_argument("-e", "--e-len", "--e_len", dest="e_len", type=int, required=True)
    parser.add_argument("--model-path", "--model_path", dest="model_path")
    parser.add_argument("--model-name", "--model_name", dest="model_name")
    parser.add_argument("--model-name-suffix", "--model_name_suffix", dest="model_name_suffix")
    parser.add_argument("--tokenizer-path", "--tokenizer_path", dest="tokenizer_path")
    parser.add_argument("--haystack-dir", default=str(Path(__file__).with_name("PaulGrahamEssays")))
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parents[2] / "work_dirs" / "needle")
    parser.add_argument("--needle", default="\n\nRemember, the best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n")
    parser.add_argument("--retrieval-question", default="what is the best thing to do in San Francisco?\n\nAnswer: The best thing to do in San Francisco is")
    parser.add_argument("--simulation-length", "--simulation_length", dest="simulation_length", type=int, default=50)
    parser.add_argument("--context-lengths-num-intervals", "--context_lengths_num_intervals", dest="context_lengths_num_intervals", type=int, default=40)
    parser.add_argument("--document-depth-percent-intervals", "--document_depth_percent_intervals", dest="document_depth_percent_intervals", type=int, default=10)
    parser.add_argument("--context-lengths-min", "--context_lengths_min", dest="context_lengths_min", type=int, default=1000)
    parser.add_argument("--context-lengths-max", "--context_lengths_max", dest="context_lengths_max", type=int, default=1048000)
    parser.add_argument("--document-depth-percent-min", "--document_depth_percent_min", dest="document_depth_percent_min", type=int, default=0)
    parser.add_argument("--document-depth-percent-max", "--document_depth_percent_max", dest="document_depth_percent_max", type=int, default=100)
    parser.add_argument("--prefilling-chunk-size", "--prefilling_chunk_size", dest="prefilling_chunk_size", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=51)
    parser.add_argument("--method", choices=METHOD_CHOICES, default="none")
    parser.add_argument("--max-capacity-prompt", "--max_capacity_prompt", dest="max_capacity_prompt", type=int, default=512)
    parser.add_argument("--window-size", "--window_size", "--recent_size", dest="window_size", type=int, default=32)
    parser.add_argument("--kernel-size", "--kernel_size", dest="kernel_size", type=int, default=7)
    parser.add_argument("--pooling", choices=("avgpool", "maxpool"), default="maxpool")
    parser.add_argument("--sink-size", "--sink_size", dest="sink_size", type=int, default=4)
    parser.add_argument("--pyramid-beta", type=float, default=0.5)
    parser.add_argument("--quest-page-size", type=int, default=16)
    parser.add_argument("--nacl-proxy-size", type=int, default=32)
    parser.add_argument("--nacl-proxy-mode", choices=("suffix", "prefix", "edges"), default="suffix")
    parser.add_argument("--nacl-random-budget", type=int, default=0)
    parser.add_argument("--scissorhands-decay", type=float, default=1.0)
    parser.add_argument("--scissorhands-selection", choices=("topk", "prob"), default="topk")
    parser.add_argument("--random-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if bool(args.model_path) == bool(args.model_name):
        raise ValueError("Provide exactly one of --model-path or --model-name")
    if args.method != "none" and args.prefilling_chunk_size is not None:
        raise ValueError("CompressedDynamicCache requires one-shot prefill; omit --prefilling-chunk-size")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    torch.manual_seed(args.seed)
    _load_backbone_registration()
    model_name = args.model_path or args.model_name
    tester = LLMNeedleHaystackTester(
        args=args,
        model_name=model_name,
        model_name_suffix=args.model_name_suffix,
        tokenizer_path=args.tokenizer_path,
        haystack_dir=args.haystack_dir,
        needle=args.needle,
        retrieval_question=args.retrieval_question,
        save_contexts=True,
        save_results=True,
        simulation_length=args.simulation_length,
        context_lengths_min=args.context_lengths_min,
        context_lengths_max=args.context_lengths_max,
        context_lengths_num_intervals=args.context_lengths_num_intervals,
        document_depth_percent_intervals=args.document_depth_percent_intervals,
        document_depth_percent_min=args.document_depth_percent_min,
        document_depth_percent_max=args.document_depth_percent_max,
    )
    tester.start_test(args)


if __name__ == "__main__":
    main()
