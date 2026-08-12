"""Compatibility exports for Transformers Llama plus OpenToMe's adapter."""

from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM,
    LlamaForQuestionAnswering, LlamaForSequenceClassification,
    LlamaForTokenClassification, LlamaModel, LlamaPreTrainedModel,
    LlamaRMSNorm, apply_rotary_pos_emb, repeat_kv,
)
from .kv_compression import patch_llama_model, unpatch_llama_model

__all__ = [
    "LlamaAttention", "LlamaDecoderLayer", "LlamaForCausalLM",
    "LlamaForQuestionAnswering", "LlamaForSequenceClassification",
    "LlamaForTokenClassification", "LlamaModel", "LlamaPreTrainedModel",
    "LlamaRMSNorm", "apply_rotary_pos_emb", "patch_llama_model", "repeat_kv",
    "unpatch_llama_model",
]
