"""Model-family dispatch for OpenToMe compressed cache adapters."""


def patch_model_for_kv_compression(model):
    model_type = getattr(model.config, "model_type", None)
    if model_type == "llama":
        from .llama import patch_llama_model

        return patch_llama_model(model)
    if model_type == "mistral":
        from .mistral import patch_mistral_model

        return patch_mistral_model(model)
    raise ValueError(f"KV compression does not support model_type={model_type!r}")


def unpatch_model_for_kv_compression(model):
    model_type = getattr(model.config, "model_type", None)
    if model_type == "llama":
        from .llama import unpatch_llama_model

        return unpatch_llama_model(model)
    if model_type == "mistral":
        from .mistral import unpatch_mistral_model

        return unpatch_mistral_model(model)
    raise ValueError(f"KV compression does not support model_type={model_type!r}")


__all__ = ["patch_model_for_kv_compression", "unpatch_model_for_kv_compression"]
