import json
import os
from typing import Sequence

from .abstract_tokenizer import Tokenizer


class DNATokenizer(Tokenizer):
    """Single-nucleotide tokenizer compatible with HyenaDNA/Caduceus ids."""

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        characters: Sequence[str] = ("A", "C", "G", "T", "N"),
        add_bos: bool = False,
        add_eos: bool = False,
        padding_side: str = "right",
        model_max_length: int | None = None,
    ):
        self.characters = tuple(ch.upper() for ch in characters)
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.padding_side = padding_side
        self.model_max_length = model_max_length

        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.bos_token = "[BOS]"
        self.mask_token = "[MASK]"
        self.pad_token = "[PAD]"
        self.reserved_token = "[RESERVED]"
        self.unk_token = "[UNK]"
        self.eos_token = self.sep_token

        self._vocab_str_to_int = {
            self.cls_token: 0,
            self.sep_token: 1,
            self.bos_token: 2,
            self.mask_token: 3,
            self.pad_token: 4,
            self.reserved_token: 5,
            self.unk_token: 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        self.cls_token_id = self._vocab_str_to_int[self.cls_token]
        self.sep_token_id = self._vocab_str_to_int[self.sep_token]
        self.bos_token_id = self._vocab_str_to_int[self.bos_token]
        self.eos_token_id = self.sep_token_id
        self.mask_token_id = self._vocab_str_to_int[self.mask_token]
        self.pad_token_id = self._vocab_str_to_int[self.pad_token]
        self.unk_token_id = self._vocab_str_to_int[self.unk_token]

        complement_map = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        self._complement_map = {}
        for token, token_id in self._vocab_str_to_int.items():
            complement = complement_map.get(token)
            self._complement_map[token_id] = (
                self._vocab_str_to_int[complement] if complement else token_id
            )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    @property
    def complement_map(self) -> dict[int, int]:
        return self._complement_map

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab_str_to_int)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def _normalize_sequence(self, text: str) -> str:
        return "".join(text.split()).upper()

    def encode(
        self,
        text: str,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"DNATokenizer.encode expects str, got {type(text)}")

        if not add_special_tokens:
            add_bos = False
            add_eos = False
        if add_bos is None:
            add_bos = self.add_bos
        if add_eos is None:
            add_eos = self.add_eos

        tokens = [
            self._vocab_str_to_int.get(ch, self.unk_token_id)
            for ch in self._normalize_sequence(text)
        ]
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def __call__(
        self,
        text,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_attention_mask: bool = True,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
        add_special_tokens: bool = True,
    ):
        texts = [text] if isinstance(text, str) else list(text)
        max_length = max_length or self.model_max_length

        batch_ids = [
            self.encode(
                item,
                add_bos=add_bos,
                add_eos=add_eos,
                add_special_tokens=add_special_tokens,
            )
            for item in texts
        ]

        if truncation and max_length is not None:
            batch_ids = [ids[:max_length] for ids in batch_ids]

        if padding:
            if padding == "max_length":
                if max_length is None:
                    raise ValueError("max_length is required when padding='max_length'")
                pad_to = max_length
            else:
                pad_to = max(len(ids) for ids in batch_ids)
            batch_ids = [self._pad(ids, pad_to) for ids in batch_ids]

        output = {"input_ids": batch_ids}
        if return_attention_mask:
            output["attention_mask"] = [
                [0 if token == self.pad_token_id else 1 for token in ids]
                for ids in batch_ids
            ]

        if return_tensors == "pt":
            import torch

            output = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}

        return output

    def _pad(self, ids: list[int], length: int) -> list[int]:
        if len(ids) >= length:
            return ids[:length]
        pad = [self.pad_token_id] * (length - len(ids))
        if self.padding_side == "left":
            return pad + ids
        return ids + pad

    def decode(self, tokens, skip_special_tokens: bool = True):
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if isinstance(tokens, int):
            tokens = [tokens]
        if tokens and isinstance(tokens[0], (list, tuple)):
            return self.batch_decode(tokens, skip_special_tokens=skip_special_tokens)

        pieces = []
        for token in tokens:
            piece = self._vocab_int_to_str.get(int(token), self.unk_token)
            if skip_special_tokens and piece.startswith("[") and piece.endswith("]"):
                continue
            pieces.append(piece)
        return "".join(pieces)

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> list[str]:
        if hasattr(sequences, "tolist"):
            sequences = sequences.tolist()
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens)
            for seq in sequences
        ]

    def get_token_offsets(
        self, text: str, tokens: list[int] | None = None
    ) -> tuple[list[str], list[int]]:
        pieces = list(self._normalize_sequence(text))
        return pieces, list(range(len(pieces)))

    @classmethod
    def from_pretrained(cls, load_directory: str):
        config_path = os.path.join(load_directory, "dna_tokenizer_config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(load_directory, "tokenizer_config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        cfg.pop("tokenizer_class", None)
        return cls(**cfg)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config = {
            "tokenizer_class": self.__class__.__name__,
            "characters": list(self.characters),
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
            "padding_side": self.padding_side,
            "model_max_length": self.model_max_length,
        }
        with open(os.path.join(save_directory, "dna_tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        return (save_directory,)
