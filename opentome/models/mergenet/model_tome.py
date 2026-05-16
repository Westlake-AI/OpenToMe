# Backward-compatibility shim: canonical file is model_a0.py
from opentome.models.mergenet.model_a0 import *  # noqa: F401,F403
from opentome.models.mergenet.model_a0 import (
    ToMEViTLocalEncoder,
    ToMELocalEncoder,
    ToMEHybridModel,
    CLSToMEHybridModel,
)
