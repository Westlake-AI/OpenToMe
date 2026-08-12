"""OpenToMe model namespace with lazy imports.

Import model families explicitly (for example ``opentome.models.llama``) to
avoid loading every optional model dependency at package import time.
"""

from importlib import import_module


_EXPORT_MODULES = {
    "BltConfig": "opentome.models.blt", "BltModel": "opentome.models.blt", "BltForCausalLM": "opentome.models.blt",
    "DeltaNetConfig": "opentome.models.delta_net", "DeltaNetForCausalLM": "opentome.models.delta_net", "DeltaNetModel": "opentome.models.delta_net",
    "GatedDeltaNetConfig": "opentome.models.gated_deltanet", "GatedDeltaNetForCausalLM": "opentome.models.gated_deltanet", "GatedDeltaNetModel": "opentome.models.gated_deltanet",
    "GLAConfig": "opentome.models.gla", "GLAForCausalLM": "opentome.models.gla", "GLAModel": "opentome.models.gla",
    "HNetConfig": "opentome.models.hnet", "HNetModel": "opentome.models.hnet", "HNetForCausalLM": "opentome.models.hnet",
    "HyenaConfig": "opentome.models.hyena", "HyenaDNAModel": "opentome.models.hyena", "HyenaDNAForCausalLM": "opentome.models.hyena", "HyenaDNAForSequenceClassification": "opentome.models.hyena",
    "TransformerConfig": "opentome.models.transformer", "TransformerForCausalLM": "opentome.models.transformer", "TransformerModel": "opentome.models.transformer",
    "MergeNetConfig": "opentome.models.mergenet_nlp", "MergeNetForCausalLM": "opentome.models.mergenet_nlp", "MergeNetModel": "opentome.models.mergenet_nlp",
    "GSAConfig": "opentome.models.gsa", "GSAForCausalLM": "opentome.models.gsa", "GSAModel": "opentome.models.gsa",
    "DeiTModel": "opentome.models.deit.deit", "deit_s": "opentome.models.deit.deit", "deit_s_extend": "opentome.models.deit.deit",
    "HybridToMeModel": "opentome.models.mergenet.model",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
