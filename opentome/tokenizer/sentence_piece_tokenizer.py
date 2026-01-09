# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
import os

try:
    from sentencepiece import SentencePieceProcessor

    has_sp = True
except ImportError:
    has_sp = False

from .abstract_tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class SentencePieceTokenizer(Tokenizer):
    def __init__(
        self, model_path: str, add_bos: bool = True, add_eos: bool = True
    ) -> None:
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        self.add_bos = add_bos
        self.add_eos = add_eos
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        # ---- HF-compatible token ids ----
        self.bos_token_id = self.bos_id
        self.eos_token_id = self.eos_id
        self.pad_token_id = self.pad_id
        self.unk_token_id = self.sp_model.unk_id()

        # ---- optional but recommended ----
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

        self.model_max_length = int(1e9)

    # ------ jinxin: 为了兼容HF的batch-wise encode ------ #
    def __call__(
        self,
        text,
        return_attention_mask: bool = False,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
    ):
        # ---- normalize to batch ----
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, list):
            if not all(isinstance(t, str) for t in text):
                raise TypeError("All elements must be str")
            texts = text
        else:
            raise TypeError(
                f"SentencePieceTokenizer expects str or List[str], got {type(text)}"
            )

        # ---- tokenize ----
        input_ids = [
            self.encode(t, add_bos=add_bos, add_eos=add_eos)
            for t in texts
        ]

        out = {"input_ids": input_ids}

        if return_attention_mask:
            out["attention_mask"] = [
                [1] * len(ids) for ids in input_ids
            ]

        return out

    def get_vocab_size(self) -> int:
        return self.n_words

    def encode(self, s: str, add_bos: bool | None = None, add_eos: bool | None = None):
        if add_bos is None:
            add_bos = self.add_bos

        if add_eos is None:
            add_eos = self.add_eos
        assert type(s) is str
        tokens = (
            [self.bos_id] * add_bos + self.sp_model.encode(s) + [self.eos_id] * add_eos
        )
        return tokens

    # def decode(self, tokens: list[int]):
    #     return self.sp_model.decode(tokens)
    
    # ------ jinxin ------ #
    def batch_decode(self, sequences, skip_special_tokens: bool = True):
        texts = []
        for seq in sequences:
            if skip_special_tokens:
                seq = [
                    t for t in seq
                    if t not in {self.bos_id, self.eos_id, self.pad_id}
                ]
            texts.append(self.decode(seq))
        return texts
    
    def decode(self, tokens: list[int], skip_special_tokens: bool = True):
        if skip_special_tokens:
            tokens = [
                t for t in tokens
                if t not in {self.bos_id, self.eos_id, self.pad_id}
            ]
        return self.sp_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: list[int] | None = None
    ) -> tuple[list[str], list[int]]:
        pieces = self.sp_model.encode_as_immutable_proto(text).pieces
        substrs = [p.surface for p in pieces]
        offsets = [p.begin for p in pieces]
        return substrs, offsets
