from ..base import KVCompressionConfig, KVCompressionPolicy
from .cam import CAMPolicy
from .h2o import H2OPolicy
from .l2norm import L2NormPolicy
from .nacl import NACLPolicy
from .pyramidkv import PyramidKVPolicy
from .quest import QuestPolicy
from .scissorhands import ScissorhandsPolicy
from .snapkv import SnapKVPolicy
from .streamingkv import StreamingKVPolicy


POLICY_REGISTRY = {
    "streamingkv": StreamingKVPolicy,
    "streamingllm": StreamingKVPolicy,
    "h2o": H2OPolicy,
    "snapkv": SnapKVPolicy,
    "pyramidkv": PyramidKVPolicy,
    "l2norm": L2NormPolicy,
    "cam": CAMPolicy,
    "quest": QuestPolicy,
    "nacl": NACLPolicy,
    "scissorhands": ScissorhandsPolicy,
}


def register_policy(name, policy_class):
    if not issubclass(policy_class, KVCompressionPolicy):
        raise TypeError("policy_class must inherit KVCompressionPolicy")
    POLICY_REGISTRY[name.lower()] = policy_class


def build_policy(config: KVCompressionConfig) -> KVCompressionPolicy:
    try:
        return POLICY_REGISTRY[config.method](config)
    except KeyError as exc:
        raise ValueError(f"No KV compression policy registered for {config.method!r}") from exc


__all__ = [
    "CAMPolicy", "H2OPolicy", "L2NormPolicy", "NACLPolicy",
    "POLICY_REGISTRY", "PyramidKVPolicy", "QuestPolicy",
    "ScissorhandsPolicy", "SnapKVPolicy", "StreamingKVPolicy",
    "build_policy", "register_policy",
]
