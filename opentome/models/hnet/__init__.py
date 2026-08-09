
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_hnet import HNetConfig
from .modeling_hnet import HNetModel, HNetForCausalLM

AutoConfig.register(HNetConfig.model_type, HNetConfig, exist_ok=True)
AutoModel.register(HNetConfig, HNetModel, exist_ok=True)
AutoModelForCausalLM.register(HNetConfig, HNetForCausalLM, exist_ok=True)


__all__ = ["HNetConfig", "HNetModel", "HNetForCausalLM"]
