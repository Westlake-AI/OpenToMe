# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
from typing import Any

from pydantic import BaseModel

from .blt_tokenizer import BltTokenizer
from .tiktoken_tokenizer import TikTokenTokenizer

try:
    from sentencepiece import SentencePieceProcessor
    has_sp = True
except ImportError:
    has_sp = False

try:
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe
    has_tiktoken = True
except ImportError:
    has_tiktoken = False

from .abstract_tokenizer import Tokenizer
from .sentence_piece_tokenizer import SentencePieceTokenizer

logger = logging.getLogger(__name__)


class TokenizerArgs(BaseModel):
    name: str = "bytes"
    init_kwargs: dict[str, Any] | None = None

    def build(self) -> Tokenizer:
        if self.init_kwargs is None:
            init_kwargs = {}
        else:
            init_kwargs = self.init_kwargs
        if self.name == "blt":
            return BltTokenizer(**init_kwargs)
        elif self.name == "mock":
            raise ValueError("TODO.")
        elif self.name == "sp":
            assert has_sp, "sentencepiece not installed"
            return SentencePieceTokenizer(**init_kwargs)
        elif self.name == "tiktoken":
            assert has_tiktoken, "tiktoken not installed"
            return TikTokenTokenizer(**init_kwargs)
        else:
            raise NotImplementedError(f"{self.name} tokenizer type is not implemented")
