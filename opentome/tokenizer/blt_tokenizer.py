# Copyright (c) Meta Platforms, Inc. and affiliates.
import re

from .abstract_tokenizer import Tokenizer
from .constants import (
    BOE_ID,
    BOS_ID,
    BPE_ID,
    BYTE_UNITS,
    EOS_ID,
    OFFSET,
    PAD_ID,
)
from .sentence_piece_tokenizer import SentencePieceTokenizer


def convert_to_bytes(s):
    # check if the output is a bytes like object of the format <0x00>
    if re.match(r"<0x[0-9a-fA-F]+>", s):
        return bytes.fromhex(s[3:-1])
    else:
        return bytes(s, "utf-8", errors="ignore")


def text2bytes_bpe_delims(
    text: str,
    *,
    bpe_tokenizer,
    bpe_id: int,
    offsetting_special_char: int,
    add_bos: bool,
    add_eos: bool,
):
    cur_bpe = bpe_tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    # merge the leading space tokens
    leading_space_tokens = []
    other_bpe_tokens = []
    leading = True
    for token in cur_bpe:
        bpe_str = bpe_tokenizer.sp_model.id_to_piece(token)
        if leading and all(c == "▁" for c in bpe_str):
            leading_space_tokens.append(bpe_str)
        else:
            leading = False
            other_bpe_tokens.append(bpe_str)
    cur_bpe_strs = ["".join(leading_space_tokens)] + other_bpe_tokens

    # Remove the '▁' characters
    bpe_strs = []
    for i, bpe_str in enumerate(cur_bpe_strs):
        if (
            len(bpe_strs) <= 1
            and all([c == " " for s in bpe_strs for c in s])
            and not all(c == "▁" for c in bpe_str)
        ):
            # Remove leading space for first non space token.
            bpe_str = bpe_str.replace("▁", "")
        elif i == 0 and all(c == "▁" for c in bpe_str):
            bpe_str = " " * (len(text) - len(text.lstrip(" ")))
        else:
            bpe_str = bpe_str.replace("▁", " ")
        if len(bpe_str) > 0:
            bpe_strs.append(bpe_str)
    ex_seq = []
    # Convert bpe tokens to bytes
    for s in bpe_strs:
        byte_chunk = convert_to_bytes(s)
        proc_chunk = [int(unit) for unit in byte_chunk]
        ex_seq.extend([bpe_id - offsetting_special_char] + proc_chunk)

    return ex_seq


class BltTokenizer(Tokenizer):
    def __init__(
        self,
        *,
        vocab_size_unit_1: int = BYTE_UNITS,
        bpe_delim: bool = False,
        bpe_tokenizer_path="/home/artidoro/tokenizers/llama_v2.tokenizer.model",
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.vocab_size_unit_1 = vocab_size_unit_1
        self.boe_id = BOE_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID
        self.bpe_id = BPE_ID
        self.bpe_tokenizer_path = bpe_tokenizer_path
        if bpe_delim:
            self.bpe_tokenizer = SentencePieceTokenizer(
                model_path=self.bpe_tokenizer_path
            )
        else:
            self.bpe_tokenizer = None
        self.bpe_delim = bpe_delim
        self.offsetting_special_char = OFFSET
        self.vocab_size_unit_1 = vocab_size_unit_1
        self.n_words = vocab_size_unit_1 + self.offsetting_special_char


         # ---- HF-compatible aliases ----
        self.bos_token_id = self.bos_id
        self.eos_token_id = self.eos_id
        self.pad_token_id = self.pad_id

        # optional but recommended
        self.unk_token_id = None

    # ------ jinxin: 为了兼容HF的batch-wise encode ------ #
    def __call__(
        self,
        text,
        return_attention_mask: bool = False,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
    ):
        """
        HF-compatible tokenizer interface.
        """

        # ---- normalize to batch ----
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, list):
            if not all(isinstance(t, str) for t in text):
                raise TypeError("All elements in text list must be str")
            texts = text
        else:
            raise TypeError(
                f"BltTokenizer.__call__ expects str or List[str], got {type(text)}"
            )

        # ---- tokenize ----
        input_ids = [
            self.encode(t, add_bos=add_bos, add_eos=add_eos)
            for t in texts
        ]

        output = {"input_ids": input_ids}

        # ---- attention mask (optional) ----
        if return_attention_mask:
            output["attention_mask"] = [
                [1] * len(ids) for ids in input_ids
            ]

        return output

    def get_vocab_size(self) -> int:
        return self.n_words

    def encode(
        self, text: str, add_bos: bool | None = None, add_eos: bool | None = None
    ):
        # ------ jinxin ------ #
        if not isinstance(text, str):
            raise TypeError(
                f"BltTokenizer.encode expects str, got {type(text)}"
            )

        if add_bos is None:
            add_bos = self.add_bos
        if add_eos is None:
            add_eos = self.add_eos

        # print(text)
        if self.bpe_delim:
            tokens = text2bytes_bpe_delims(
                text,
                bpe_tokenizer=self.bpe_tokenizer,
                bpe_id=self.bpe_id,
                offsetting_special_char=self.offsetting_special_char,
                add_bos=False,
                add_eos=False,
            )
        else:
            tokens = bytes(text, encoding="utf-8", errors="ignore")

        # Offsetting
        tokens = [int(unit) + self.offsetting_special_char for unit in tokens]

        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    # ------ jinxin ------ #
    def _decode_single(self, tokens: list[int], skip_special_tokens: bool = True):
        byte_vals = []
        for tok in tokens:
            if skip_special_tokens and tok < self.offsetting_special_char:
                continue
            val = tok - self.offsetting_special_char
            if val >= 0:
                byte_vals.append(val)
        return bytes(byte_vals).decode("utf-8", errors="ignore")
    
    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
    ):
        if not isinstance(token_ids, (list, tuple)):
            raise TypeError(
                f"decode expects List[int], got {type(token_ids)}"
            )

        return self._decode_single(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def batch_decode(
        self,
        sequences,
        skip_special_tokens: bool = True,
    ):
        if not isinstance(sequences, (list, tuple)):
            raise TypeError("batch_decode expects a list of token sequences")

        return [
            self._decode_single(seq, skip_special_tokens=skip_special_tokens)
            for seq in sequences
        ]


    # def decode(self, tokens: list[int], cut_at_eos: bool = False):
    #     if cut_at_eos:
    #         for k, t in enumerate(tokens):
    #             if t == self.eos_id:
    #                 tokens = tokens[: k + 1]
    #                 break
    #     return bytes(
    #         [
    #             tok - self.offsetting_special_char
    #             for tok in tokens
    #             if tok - self.offsetting_special_char >= 0
    #         ]
    #     ).decode("utf-8", errors="ignore")

    def get_token_offsets(self, text: str, tokens: list[int] | None = None):
        # TODO: Figure out what this does
        raise NotImplementedError()
