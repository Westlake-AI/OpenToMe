
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_qwen3_next import Qwen3NextConfig
from .modeling_qwen3_next import Qwen3NextModel, Qwen3NextForCausalLM

AutoConfig.register(Qwen3NextConfig.model_type, Qwen3NextConfig, exist_ok=True)
AutoModel.register(Qwen3NextConfig, Qwen3NextModel, exist_ok=True)
AutoModelForCausalLM.register(Qwen3NextConfig, Qwen3NextForCausalLM, exist_ok=True)

__all__ = ["Qwen3NextConfig", "Qwen3NextModel", "Qwen3NextForCausalLM"]
