"""
OpenToMe Optimizers Package

Optimizer implementations aligned with SAC/opt.
"""

# SGG (Scaling with Gradient Grouping) Optimizers
from .sgg.adamw_sgg import AdamWSGG
from .sgg.adafactor_sgg import AdafactorSGG
from .sgg.lamb_sgg import LambSGG
from .sgg.shampoo_sgg import ShampooSGG

# SAC (Structured Adaptive Computation) Optimizers
from .sac.adamw_sac import AdamWSAC
from .sac.adam_mini_sac import Adam_miniSAC
from .sac.shampoo_sac import ShampooSAC
from .sac.lion_sac import SACLion

# Standard & Third-party Optimizers (aligned with SAC/opt)
from .adam_mini import Adam_mini
from .lamb import Lamb
from .shampoo import Shampoo
from .galore_adamw import AdamW as GaLore_AdamW
from .galore_adafactor import Adafactor as GaLoreAdafactor
from .svd_projector import GaLoreProjector
from .random_projector import GradientProjector
from .adan import Adan
from .apollo import AdamW as APOLLO_AdamW
from .came import CAME
from .conda import Conda, CondaProjector
from .lion import Lion
from .moga import MOGASGD
from .mars import MARS
from .muon import Muon
from .muon_ablation import MuonOldScale, MuonS4, MuonC1b, MuonBest
from .nadam import NAdamLegacy as NAdam
from .radam import RAdamLegacy as RAdam
from .sophia import SophiaG
from .soap import SOAP
from .scale import SCALE
from .rmnp import RMNP
from .adabelief import AdaBelief
from .adamp import AdamP
from .adamw import AdamWLegacy
from .adopt import Adopt
from .kron import Kron
from .laprop import LaProp
from .lars import LARS
from .nvnovograd import NvNovoGrad
from .prodigy import Prodigy

# Optional (requires bitsandbytes)
try:
    from .galore_adamw8bit import AdamW8bit as GaLoreAdamW8bit
except (ImportError, ModuleNotFoundError):
    GaLoreAdamW8bit = None

try:
    from .q_apollo import AdamW as QAPOLLOAdamW
except (ImportError, ModuleNotFoundError):
    QAPOLLOAdamW = None

__all__ = [
    # SGG
    'AdamWSGG', 'AdafactorSGG', 'LambSGG', 'ShampooSGG',
    # SAC
    'AdamWSAC', 'Adam_miniSAC', 'ShampooSAC', 'SACLion',
    # Standard
    'Adam_mini', 'Lamb', 'Shampoo', 'GaLore_AdamW', 'GaLoreAdafactor',
    'GaLoreProjector', 'GradientProjector', 'CondaProjector',
    'Adan', 'APOLLO_AdamW', 'CAME', 'Conda', 'Lion', 'MOGASGD', 'MARS', 'Muon',
    'MuonOldScale', 'MuonS4', 'MuonC1b', 'MuonBest',
    'NAdam', 'RAdam', 'SophiaG', 'SOAP', 'SCALE', 'RMNP',
    # New from SAC
    'AdaBelief', 'AdamP', 'AdamWLegacy', 'Adopt', 'Kron',
    'LaProp', 'LARS', 'NvNovoGrad', 'Prodigy',
]
