
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_mergenet import MergeNetConfig
from .model import (
    MergeNetPreTrainedModel,
    MergeNetModel,
    MergeNetForCausalLM,
    SharedLocalTransformer,
    LocalEncoderNLP,
    LatentModel,
    LocalDecoder,
    MyCrossAttention,
)

AutoConfig.register(MergeNetConfig.model_type, MergeNetConfig, exist_ok=True)
AutoModel.register(MergeNetConfig, MergeNetModel, exist_ok=True)
AutoModelForCausalLM.register(MergeNetConfig, MergeNetForCausalLM, exist_ok=True)

__all__ = [
    "MergeNetConfig",
    "MergeNetPreTrainedModel",
    "MergeNetModel",
    "MergeNetForCausalLM",
    "SharedLocalTransformer",
    "LocalEncoderNLP",
    "LatentModel",
    "LocalDecoder",
    "MyCrossAttention",
]


