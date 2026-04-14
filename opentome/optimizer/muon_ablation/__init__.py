"""Muon ablation variants for OpenToMe.

Validated variants from SAC Muon ablation study + OldScale comparison.

  MuonOldScale  — old scaling: 0.2 * sqrt(max(rows, cols)); lr=1e-3
  MuonS4        — symmetric aspect-ratio scaling; SAC best (+0.080 PPL); lr=6e-3
  MuonC1b       — Nesterov AFTER NS (orthogonal space); SAC best (+0.040 PPL); lr=6e-3
  MuonBest      — S4 + C1b combined; expected +0.12 PPL; lr=6e-3
"""

from .muon_oldscale import MuonOldScale
from .muon_s4 import MuonS4
from .muon_c1b import MuonC1b
from .muon_best import MuonBest

__all__ = ["MuonOldScale", "MuonS4", "MuonC1b", "MuonBest"]
