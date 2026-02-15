"""
OpenToMe Optimizers Package

This package contains various state-of-the-art optimizers for deep learning,
including adaptive learning rate optimizers, second-order methods, and 
specialized optimizers for large language models.

Optimizers are organized into three main categories:
- SGG (Scaling with Gradient Grouping) variants: Enhanced optimizers with gradient processing
- SAC (Structured Adaptive Computation) variants: Optimizers with structured computation
- Standard/Third-party optimizers: Original implementations and other adaptive methods
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

# Standard & Third-party Optimizers
from .adam_mini import Adam_mini
from .lamb import Lamb
from .shampoo import Shampoo
from .galore_adamw import GaLore_AdamW
from .svd_projector import GaLoreProjector
from .random_projector import GradientProjector
from .adan import Adan
from .apollo import APOLLO_AdamW
from .came import CAME, GradientProjector as CAMEGradientProjector
from .conda import Conda, CondaProjector
from .lion import Lion
from .mars import MARS
from .muon import Muon
from .nadam import NAdam
from .radam import RAdam, PlainRAdam, AdamW as RAdamW
from .sophia import SophiaG
from .soap import SOAP

__all__ = [
    # SGG Optimizers
    'AdamWSGG', 'AdafactorSGG', 'LambSGG', 'ShampooSGG',
    
    # SAC Optimizers
    'AdamWSAC', 'Adam_miniSAC', 'ShampooSAC',
    
    # Standard & Third-party Optimizers
    'Adam_mini', 'NAdam', 'RAdam', 'PlainRAdam', 'RAdamW', 'Lamb', 'Shampoo',
    'GaLore_AdamW', 'GaLoreProjector', 'GradientProjector', 'CAMEGradientProjector', 'CondaProjector',
    'Adan', 'APOLLO_AdamW','CAME', 'Conda', 'Lion', 'MARS', 'Muon', 'SophiaG', 'SOAP',
]


# SGG (Scaling with Gradient Grouping) Optimizers
SGG_OPTIMIZERS = [
    'AdamWSGG', 'AdafactorSGG', 'LambSGG', 'ShampooSGG',
]

# SAC (Structured Adaptive Computation) Optimizers
SAC_OPTIMIZERS = [
    'AdamWSAC', 'Adam_miniSAC', 'ShampooSAC',
]

# Standard & Third-party Optimizers
STANDARD_OPTIMIZERS = [
    'Adam_mini', 'Lamb', 'Shampoo', 'Adan', 'APOLLO_AdamW', 'Lion',
    'MARS', 'Muon', 'NAdam', 'RAdam', 'PlainRAdam', 'SophiaG', 'SOAP',
]

# Gradient compression and low-rank optimizers
LOW_RANK_OPTIMIZERS = [
    'GaLore_AdamW', 'CAME', 'Conda',
]

# Utility classes
UTILITY_CLASSES = [
    'GaLoreProjector', 'GradientProjector', 'CAMEGradientProjector', 'CondaProjector', 'RAdamW',
]

# Legacy categorization (for backward compatibility)
ADAM_OPTIMIZERS = SGG_OPTIMIZERS[:1] + SAC_OPTIMIZERS[:1] + ['Adam_mini']
ADAFACTOR_OPTIMIZERS = [opt for opt in SGG_OPTIMIZERS if 'Adafactor' in opt]
LAMB_OPTIMIZERS = [opt for opt in SGG_OPTIMIZERS if 'Lamb' in opt]
SECOND_ORDER_OPTIMIZERS = [opt for opt in SGG_OPTIMIZERS + SAC_OPTIMIZERS if 'Shampoo' in opt]
OTHER_OPTIMIZERS = STANDARD_OPTIMIZERS[3:]  # Exclude Adam_mini, Lamb, Shampoo