"""Shared LR scaling helpers for Muon ablation variants.

Three scaling strategies compared:

  OldScale  : 0.2 * sqrt(max(rows, cols))   — absolute size, from OldSAC/opt/muon.py
  A6 (base) : max(1, rows/cols)^0.5          — asymmetric aspect-ratio (Muon default)
  S4        : (max(r,c)/min(r,c))^0.5        — symmetric aspect-ratio (SAC best)
"""

import math


def _scale_old(update):
    """OldSAC scaling: 0.2 * sqrt(max(rows, cols)).

    Returns the multiplier only; caller does: effective_lr = lr * _scale_old(update).
    Weight decay continues to use the bare lr (preserves original asymmetry).
    """
    r, c = update.shape[-2], update.shape[-1]
    return 0.2 * math.sqrt(max(r, c))


def _scale_s4(update):
    """S4 symmetric aspect-ratio scaling: (max(r,c) / min(r,c))^0.5.

    Unlike A6 which only boosts tall matrices, S4 boosts both tall AND wide matrices.
    Key case: FFN up-proj [h, 4h] (wide matrix) — A6 scale=1.0, S4 scale=2.0.
    SAC ablation result: PPL 16.128 vs A6=16.208 (+0.080 improvement).
    """
    r, c = update.shape[-2], update.shape[-1]
    return (max(r, c) / min(r, c)) ** 0.5
