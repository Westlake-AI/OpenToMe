# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import io
import os
import tempfile
from datetime import timedelta

import fla  # noqa
import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torchtitan.tools.logging import init_logger, logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ------ jinxin added ------ #
backbone = os.environ.get("BACKBONE", "None")
print("*" * 50)
print(backbone)
# One-time import to register all custom model types with HF Auto classes.
import opentome.models  # noqa: F401
tokenizer_name = os.environ.get("TOKENIZER_NAME", "default")
print(f"Tokenizer name: {tokenizer_name}")
print("*" * 50)

# from ipdb import set_trace as point
from opentome.tokenizer.build_tokenizer import TokenizerArgs

# ------ End of jinxin added ------ #

@torch.inference_mode()
def save_pretrained(
    path: str,
    step: int,
    config: str,
    tokenizer: str
):
    logger.info(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)

    logger.info(f"Saving the config to {path}")
    config.save_pretrained(path)
    logger.info(f"Loading the tokenizer from {tokenizer}")
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    # ------ jinxin ------ #
    if tokenizer_name =="default":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True,)
    else:
        assert tokenizer_name in ["bytes", "sentencepiece", "tiktoken", "blt"], f"Invalid tokenizer name: {tokenizer_name}"
        tokenizer = TokenizerArgs(
            name=tokenizer_name,
            init_kwargs={
                "vocab_size_unit_1": 256,
                "bpe_delim": False,
                "bpe_tokenizer_path": tokenizer,
                "add_bos": True,
                "add_eos": True,
            }
        ).build()
    logger.info(f"Saving the tokenizer to {path}")
    tokenizer.save_pretrained(path)

    if tokenizer_name =="default":
        config.vocab_size = max(tokenizer.vocab_size, config.vocab_size)
    else:
        assert tokenizer_name in ["bytes", "sentencepiece", "tiktoken", "blt"], f"Invalid tokenizer name: {tokenizer_name}"
        config.vocab_size = tokenizer.get_vocab_size()

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(path, f'checkpoint/step-{step}')
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)

        logger.info(f"Initializing the model from config\n{config}")
        model = AutoModelForCausalLM.from_config(config)
        logger.info(model)
        logger.info("Loading state dict from the checkpoint")

        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        # torch.load now with default weights_only=True will work
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])

        logger.info(f"Saving the model to {path}")
        model.save_pretrained(path)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser("Convert DCP format model weights to huggingface-style.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    args = parser.parse_args()
    save_pretrained(args.path, args.step, args.config, args.tokenizer)
