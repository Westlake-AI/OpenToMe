from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification

from .configuration_hyena import HyenaConfig
from .modeling_hyena import HyenaDNAForCausalLM, HyenaDNAModel, HyenaDNAForSequenceClassification
from .tokenization_hyena import HyenaDNATokenizer

AutoConfig.register(HyenaConfig.model_type, HyenaConfig, exist_ok=True)
AutoModel.register(HyenaConfig, HyenaDNAModel, exist_ok=True)
AutoModelForCausalLM.register(HyenaConfig, HyenaDNAForCausalLM, exist_ok=True)
AutoModelForSequenceClassification.register(HyenaConfig, HyenaDNAForSequenceClassification, exist_ok=True)

__all__ = [
    "HyenaConfig",
    "HyenaDNAModel",
    "HyenaDNAForCausalLM",
    "HyenaDNAForSequenceClassification",
    "HyenaDNATokenizer",
]
