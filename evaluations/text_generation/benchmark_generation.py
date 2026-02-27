# Copyright (c) 2023-2024, Songlin Yang, Yu Zhang.

import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import fla  # noqa
from opentome.tokenizer.build_tokenizer import TokenizerArgs


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation benchmarking")
    parser.add_argument("--path", type=str, default="fla-hub/transformer-1.3B-100B")
    parser.add_argument("--model", type=str, default="transformer++", choices=["transformer++", "gsa", "gla", "delta_net", "gated_deltanet", "blt", "mergenet"])
    parser.add_argument("--tokenizer", type=str, default="default", choices=["blt", "default"])
    parser.add_argument("--prompt", type=str, default="Please introduce Westlake University in 100 words")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=256)
    parser.add_argument("--no-cache", action='store_true')
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--topp", type=float, default=0.2)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--output-generation", action='store_true')
    parser.add_argument("--compile", action='store_true')
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # --- Step 1. Load model --- #
    print(f"{args.model}")
    if args.model == "transformer++":
        import opentome.models.transformer
    elif args.model == "gsa":
        import opentome.models.gsa
    elif args.model == "gla":
        import opentome.models.gla
    elif args.model == "delta_net":
        import opentome.models.delta_net
    elif args.model == "gated_deltanet":
        import opentome.models.gated_deltanet
    elif args.model == "blt":
        import opentome.models.blt
    elif args.model == "mergenet":
        import opentome.models.mergenet_nlp

    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        device_map={"": device},
        torch_dtype=dtype,
        use_cache=not args.no_cache,
    )
    if args.compile:
        print("Compiling the model")
        model = torch.compile(model)
    model.eval()
    print(f"{model.config}\n{model}\nNumber of parameters: {model.num_parameters()} ({sizeof_fmt(model.num_parameters())})\n")


    # --- Step 2. Load tokenizer --- #
    print(f"Loading {args.path}")
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.path,
    #     trust_remote_code=True,
    #     add_eos_token=False,
    # )
    if args.tokenizer =="default":
        tokenizer = AutoTokenizer.from_pretrained(
            args.path,
            trust_remote_code=True,
            add_eos_token=False,
        )
    else:
        assert args.tokenizer in ["bytes", "sentencepiece", "tiktoken", "blt"], f"Invalid tokenizer name: {args.tokenizer}"
        tokenizer = TokenizerArgs(
            name=args.tokenizer,
            init_kwargs={
                "vocab_size_unit_1": 256,
                "bpe_delim": False,
                "bpe_tokenizer_path": args.path,
                "add_bos": True,
                "add_eos": True,
            }
        ).build()
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Step 3. Load prompt --- #
    prompt = args.prompt if args.prompt else "I'm Songlin Yang."
    tokens = tokenizer(prompt, return_tensors="pt")
    if args.tokenizer == "blt":
        input_ids = tokens["input_ids"].to(device=device)[:, :args.length].contiguous()
    else:
        input_ids = tokens.input_ids.to(device=device)[:, :args.length].contiguous()
    max_length = input_ids.shape[1] + args.maxlen

    torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode():
        text = model.generate(
            input_ids=input_ids,
            use_cache=not args.no_cache,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.bos_token_id,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.topp,
            repetition_penalty=args.repetition_penalty,
        )
    torch.cuda.synchronize()
    elapsed = time.time() - start
    if args.output_generation:
        print(f"Prompt:\n{tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0].strip()}\n")
        print(f"Generated:\n{tokenizer.batch_decode(text, skip_special_tokens=True)[0].strip()}\n")
    print(f"Prompt length: {len(input_ids[0])}, generation length: {len(text[0]) - len(input_ids[0])}")
    print(f"Total prompt processing + decoding time: {elapsed * 1000:.0f}ms")
    print(f"Max memory used: {sizeof_fmt(torch.cuda.max_memory_allocated())}")
