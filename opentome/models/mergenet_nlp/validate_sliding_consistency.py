"""
Validation script focused on sliding-mode consistency.

What this script checks:
1) LoD causal/window mask rule correctness.
2) Train-style vs infer-style decoder bias equivalence.
3) LoD batch decode vs step decode equivalence (same KV queue).
4) LoT/LaM cache equivalence (full sequence vs token-by-token cache).

This intentionally does NOT compare with full_recompute generation.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

# Ensure repo root is on PYTHONPATH when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from opentome.models.mergenet_nlp import MergeNetConfig, MergeNetForCausalLM
except Exception:
    # Allow running this file directly from its own directory.
    from configuration_mergenet import MergeNetConfig
    from model import MergeNetForCausalLM


@dataclass
class CheckResult:
    name: str
    passed: bool
    metric: float
    threshold: float
    note: str = ""


def _device_dtype() -> Tuple[torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def _make_model(device: torch.device, dtype: torch.dtype) -> MergeNetForCausalLM:
    # Keep this model small for fast, deterministic checks.
    config = MergeNetConfig(
        vocab_size=320,
        hidden_size=128,
        num_local_layers=2,
        local_depth=2,
        num_latent_layers=2,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=256,
        max_position_embeddings=2048,
        lambda_local=4.0,
        dtem_window_size=4,
        dtem_t=1,
        use_softkmax=False,
        phase="phase2",
        W_infer=4,
    )
    model = MergeNetForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()
    return model


def check_mask_rule_correctness(model: MergeNetForCausalLM, device: torch.device, dtype: torch.dtype) -> CheckResult:
    """
    Verify visibility rule:
      visible iff floor(t/lambda)-W_infer <= j <= floor(t/lambda)
    """
    torch.manual_seed(0)
    L = 40
    N = 20
    q = torch.zeros(1, L, model.config.hidden_size, device=device, dtype=dtype)
    kv = torch.zeros(1, N, model.config.hidden_size, device=device, dtype=dtype)
    byte_positions = torch.arange(L, device=device, dtype=torch.long)
    latent_positions = torch.arange(N, device=device, dtype=torch.long)

    with torch.no_grad():
        bias = model.model.local_decoder.build_decoder_bias(
            q, kv, byte_positions=byte_positions, latent_positions=latent_positions
        )[0].float()

    lambda_local = model.config.lambda_local
    w_infer = model.config.W_infer
    violations = 0
    total = 0
    invalid_cutoff = -1e9
    for t in range(L):
        center = math.floor(t / lambda_local)
        left = center - w_infer
        right = center
        for j in range(N):
            total += 1
            should_visible = (left <= j <= right)
            is_visible = bias[t, j].item() > invalid_cutoff
            if should_visible != is_visible:
                violations += 1

    violation_ratio = violations / max(total, 1)
    return CheckResult(
        name="mask_rule_correctness",
        passed=(violations == 0),
        metric=violation_ratio,
        threshold=0.0,
        note=f"violations={violations}/{total}",
    )


def check_train_infer_bias_equivalence(model: MergeNetForCausalLM, device: torch.device, dtype: torch.dtype) -> CheckResult:
    """
    Compare train-style row extraction vs infer-style single-step call
    under same absolute positions and latent queue positions.
    """
    torch.manual_seed(1)
    L = 24
    Nq = model.config.W_infer + 1
    start_j = 3
    latent_positions = torch.arange(start_j, start_j + Nq, device=device, dtype=torch.long)

    q_train = torch.zeros(1, L, model.config.hidden_size, device=device, dtype=dtype)
    kv = torch.zeros(1, Nq, model.config.hidden_size, device=device, dtype=dtype)
    byte_positions = torch.arange(L, device=device, dtype=torch.long)
    t_star = 17

    with torch.no_grad():
        bias_train_all = model.model.local_decoder.build_decoder_bias(
            q_train, kv, byte_positions=byte_positions, latent_positions=latent_positions
        )
        bias_train_row = bias_train_all[:, t_star:t_star + 1, :]

        q_step = torch.zeros(1, 1, model.config.hidden_size, device=device, dtype=dtype)
        bias_infer = model.model.local_decoder.build_decoder_bias(
            q_step,
            kv,
            byte_positions=torch.tensor([t_star], device=device, dtype=torch.long),
            latent_positions=latent_positions,
        )

    max_abs_diff = (bias_train_row.float() - bias_infer.float()).abs().max().item()
    tol = 1e-6
    return CheckResult(
        name="train_infer_bias_equivalence",
        passed=max_abs_diff <= tol,
        metric=max_abs_diff,
        threshold=tol,
    )


def check_lod_batch_step_equivalence(model: MergeNetForCausalLM, device: torch.device, dtype: torch.dtype) -> CheckResult:
    """
    Under fixed latent KV queue, LoD batch decoding must match step decoding.
    """
    torch.manual_seed(2)
    L = 18
    Nq = model.config.W_infer + 1
    hidden = model.config.hidden_size
    query = torch.randn(1, L, hidden, device=device, dtype=dtype)
    kv = torch.randn(1, Nq, hidden, device=device, dtype=dtype)
    byte_positions = torch.arange(30, 30 + L, device=device, dtype=torch.long)
    latent_positions = torch.arange(4, 4 + Nq, device=device, dtype=torch.long)

    with torch.no_grad():
        out_batch = model.model.local_decoder(
            query,
            kv,
            byte_positions=byte_positions,
            latent_positions=latent_positions,
        )

        step_outs = []
        for i in range(L):
            out_i = model.model.local_decoder(
                query[:, i:i + 1, :],
                kv,
                byte_positions=byte_positions[i:i + 1],
                latent_positions=latent_positions,
            )
            step_outs.append(out_i)
        out_step = torch.cat(step_outs, dim=1)

    max_abs_diff = (out_batch.float() - out_step.float()).abs().max().item()
    tol = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    return CheckResult(
        name="lod_batch_step_equivalence",
        passed=max_abs_diff <= tol,
        metric=max_abs_diff,
        threshold=tol,
    )


def _lot_cache_equivalence(model: MergeNetForCausalLM, input_ids: torch.Tensor) -> float:
    with torch.no_grad():
        full = model.model.shared_local_transformer(input_ids, attention_mask=None, use_cache=False)
        past = None
        step_h = []
        for i in range(input_ids.shape[1]):
            h_i, past = model.model.shared_local_transformer(
                input_ids[:, i:i + 1],
                attention_mask=None,
                past_key_values=past,
                use_cache=True,
            )
            step_h.append(h_i)
        step = torch.cat(step_h, dim=1)
    return (full.float() - step.float()).abs().max().item()


def _lam_cache_equivalence(model: MergeNetForCausalLM, latent_words: torch.Tensor) -> float:
    with torch.no_grad():
        full = model.model.latent_model(latent_words, attention_mask=None, use_cache=False)
        past = None
        step_h = []
        for i in range(latent_words.shape[1]):
            h_i, past = model.model.latent_model(
                latent_words[:, i:i + 1, :],
                attention_mask=None,
                past_key_values=past,
                use_cache=True,
            )
            step_h.append(h_i)
        step = torch.cat(step_h, dim=1)
    return (full.float() - step.float()).abs().max().item()


def check_cache_equivalence(model: MergeNetForCausalLM, device: torch.device, dtype: torch.dtype) -> Tuple[CheckResult, CheckResult]:
    torch.manual_seed(3)
    seq_len = 20
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    max_diff_lot = _lot_cache_equivalence(model, input_ids)

    latent_len = 12
    latent_words = torch.randn(1, latent_len, model.config.hidden_size, device=device, dtype=dtype)
    max_diff_lam = _lam_cache_equivalence(model, latent_words)

    tol = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    return (
        CheckResult("lot_cache_equivalence", max_diff_lot <= tol, max_diff_lot, tol),
        CheckResult("lam_cache_equivalence", max_diff_lam <= tol, max_diff_lam, tol),
    )


def check_full_path_causality(model: MergeNetForCausalLM, device: torch.device, dtype: torch.dtype) -> Tuple[CheckResult, CheckResult]:
    """
    Future perturbation tests on full-path forward only:
      - LoT: perturb future input_ids, check prefix hidden stays unchanged.
      - LaM: perturb future latent_words, check prefix hidden stays unchanged.
    """
    torch.manual_seed(4)

    # LoT causality
    seq_len = 32
    prefix_len = 16
    base_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    alt_ids = base_ids.clone()
    alt_ids[:, prefix_len:] = torch.randint(
        0, model.config.vocab_size, (1, seq_len - prefix_len), device=device
    )
    with torch.no_grad():
        h_base = model.model.shared_local_transformer(base_ids, attention_mask=None, use_cache=False)
        h_alt = model.model.shared_local_transformer(alt_ids, attention_mask=None, use_cache=False)
    lot_prefix_diff = (h_base[:, :prefix_len, :].float() - h_alt[:, :prefix_len, :].float()).abs().max().item()

    # LaM causality
    latent_len = 24
    latent_prefix = 12
    z_base = torch.randn(1, latent_len, model.config.hidden_size, device=device, dtype=dtype)
    z_alt = z_base.clone()
    z_alt[:, latent_prefix:, :] = torch.randn(1, latent_len - latent_prefix, model.config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        o_base = model.model.latent_model(z_base, attention_mask=None, use_cache=False)
        o_alt = model.model.latent_model(z_alt, attention_mask=None, use_cache=False)
    lam_prefix_diff = (o_base[:, :latent_prefix, :].float() - o_alt[:, :latent_prefix, :].float()).abs().max().item()

    tol = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    return (
        CheckResult("lot_full_path_causality", lot_prefix_diff <= tol, lot_prefix_diff, tol),
        CheckResult("lam_full_path_causality", lam_prefix_diff <= tol, lam_prefix_diff, tol),
    )


def _print_result(res: CheckResult):
    status = "PASS" if res.passed else "FAIL"
    extra = f" ({res.note})" if res.note else ""
    print(f"[{status}] {res.name}: metric={res.metric:.6g}, threshold={res.threshold:.6g}{extra}")


def main():
    device, dtype = _device_dtype()
    print("=" * 70)
    print("Sliding Consistency Validation")
    print("=" * 70)
    print(f"device={device}, dtype={dtype}")

    model = _make_model(device, dtype)

    results = []
    results.append(check_mask_rule_correctness(model, device, dtype))
    results.append(check_train_infer_bias_equivalence(model, device, dtype))
    results.append(check_lod_batch_step_equivalence(model, device, dtype))
    lot_res, lam_res = check_cache_equivalence(model, device, dtype)
    results.extend([lot_res, lam_res])
    lot_causal_res, lam_causal_res = check_full_path_causality(model, device, dtype)
    results.extend([lot_causal_res, lam_causal_res])

    print("-" * 70)
    for res in results:
        _print_result(res)

    passed = sum(1 for r in results if r.passed)
    print("-" * 70)
    print(f"Summary: {passed}/{len(results)} checks passed.")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

