# Copyright (c) Meta Platforms, Inc. and affiliates.
import re
import json
import os
import torch

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
        offsetting_special_char: int = OFFSET
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
        # self.offsetting_special_char = OFFSET
        self.offsetting_special_char = offsetting_special_char
        self.vocab_size_unit_1 = vocab_size_unit_1
        self.n_words = vocab_size_unit_1 + self.offsetting_special_char

        self.padding_side = "right"

        # ---- HF-compatible aliases ----
        self.bos_token_id = self.bos_id
        self.eos_token_id = self.eos_id
        self.pad_token_id = self.pad_id

        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

        self.unk_token_id = None
        self.unk_token = None

        # optional but recommended
        self.unk_token_id = None

    def __call__(
        self,
        text,
        padding=False,
        truncation=False,
        return_tensors=None,
        return_attention_mask=True,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
    ):
        # normalize batch
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        batch_ids = [
            self.encode(t, add_bos=add_bos, add_eos=add_eos)
            for t in texts
        ]

        # ---- padding ----
        if padding:
            max_len = max(len(x) for x in batch_ids)

            if self.padding_side == "right":
                batch_ids = [
                    x + [self.pad_token_id] * (max_len - len(x))
                    for x in batch_ids
                ]
            else:
                batch_ids = [
                    [self.pad_token_id] * (max_len - len(x)) + x
                    for x in batch_ids
                ]

        # ---- attention mask ----
        if return_attention_mask:
            attn = [
                [1 if t != self.pad_token_id else 0 for t in seq]
                for seq in batch_ids
            ]

        out = {"input_ids": batch_ids}
        if return_attention_mask:
            out["attention_mask"] = attn

        # ---- tensor ----
        if return_tensors == "pt":
            out = {
                k: torch.tensor(v, dtype=torch.long)
                for k, v in out.items()
            }

        return out

    def get_vocab_size(self) -> int:
        return self.n_words
    
    def vocab_size(self) -> int:
        return self.get_vocab_size()
    
    
    @classmethod
    def from_pretrained(cls, load_directory):

        with open(
            os.path.join(load_directory, "tokenizer_config.json")
        ) as f:
            cfg = json.load(f)

        if cfg.get("bpe_delim", False):
            cfg["bpe_tokenizer_path"] = os.path.join(
                load_directory, cfg["bpe_tokenizer_path"]
            )

        return cls(**cfg)


    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        config = {
            "vocab_size_unit_1": self.vocab_size_unit_1,
            "bpe_delim": self.bpe_delim,
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
            "offsetting_special_char": self.offsetting_special_char,
        }

        if self.bpe_delim:
            config["bpe_tokenizer_path"] = self.bpe_tokenizer_path

        with open(
            os.path.join(save_directory, "tokenizer_config.json"), "w"
        ) as f:
            json.dump(config, f, indent=2)

        if self.bpe_delim:
            import shutil
            shutil.copy(
                self.bpe_tokenizer_path,
                os.path.join(save_directory, "tokenizer.model"),
            )

        print(f"Tokenizer saved to {save_directory}")

    def encode(
        self,
        text: str,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
        add_special_tokens: bool = True,
    ):
        if not isinstance(text, str):
            raise TypeError(
                f"BltTokenizer.encode expects str, got {type(text)}"
            )

        if add_special_tokens is False:
            add_bos = False
            add_eos = False

        if add_bos is None:
            add_bos = self.add_bos
        if add_eos is None:
            add_eos = self.add_eos

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
    
    def decode(self, token_ids, skip_special_tokens: bool = True):
        # tensor -> list
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # single id
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        # batch passed
        if (
            isinstance(token_ids, (list, tuple))
            and len(token_ids) > 0
            and isinstance(token_ids[0], (list, tuple))
        ):
            return self.batch_decode(
                token_ids, skip_special_tokens=skip_special_tokens
            )

        if not isinstance(token_ids, (list, tuple)):
            raise TypeError(
                f"decode expects int or List[int], got {type(token_ids)}"
            )

        return self._decode_single(
            token_ids, skip_special_tokens=skip_special_tokens
        )


    def batch_decode(
        self,
        sequences,
        skip_special_tokens: bool = True,
    ):
        # ---- HF compatibility ----

        # tensor -> list
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()

        if not isinstance(sequences, (list, tuple)):
            raise TypeError(
                f"batch_decode expects Tensor or List[List[int]], got {type(sequences)}"
            )

        return [
            self._decode_single(seq, skip_special_tokens=skip_special_tokens)
            for seq in sequences
        ]


    def get_token_offsets(self, text: str, tokens: list[int] | None = None):
        # TODO: Figure out what this does
        raise NotImplementedError()
