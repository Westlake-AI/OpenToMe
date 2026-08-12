from .base import KVCompressionConfig, KVCompressionPolicy
from .cache import CompressedDynamicCache
from .methods import (
    CAMPolicy,
    H2OPolicy,
    L2NormPolicy,
    NACLPolicy,
    POLICY_REGISTRY,
    PyramidKVPolicy,
    QuestPolicy,
    ScissorhandsPolicy,
    SnapKVPolicy,
    StreamingKVPolicy,
    build_policy,
    register_policy,
)
from .methods.selectors import compress_minicache_pair, restore_minicache_pair


def build_compressed_cache(method: str, **kwargs) -> CompressedDynamicCache:
    return CompressedDynamicCache(KVCompressionConfig(method=method, **kwargs))


__all__ = [
    "CAMPolicy", "CompressedDynamicCache", "H2OPolicy", "KVCompressionConfig",
    "KVCompressionPolicy", "L2NormPolicy", "NACLPolicy", "POLICY_REGISTRY",
    "PyramidKVPolicy", "QuestPolicy", "ScissorhandsPolicy", "SnapKVPolicy",
    "StreamingKVPolicy", "build_compressed_cache", "build_policy",
    "compress_minicache_pair", "register_policy", "restore_minicache_pair",
]
