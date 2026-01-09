
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_blt import BltConfig
from .modeling_blt import BltModel, BltForCausalLM

AutoConfig.register(BltConfig.model_type, BltConfig, exist_ok=True)
AutoModel.register(BltConfig, BltModel, exist_ok=True)
AutoModelForCausalLM.register(BltConfig, BltForCausalLM, exist_ok=True)

__all__ = ["BltConfig", "BltModel", "BltForCausalLM"]
