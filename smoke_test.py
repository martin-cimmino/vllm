#!/usr/bin/env python3
"""Minimal vLLM smoke-test loader.

Loads a vLLM `LLM` instance and generates a short completion to validate
that the package, tokenizer, and GPU are usable. Designed to be short and
idempotent for CI / quick cluster checks.

Usage:
  python validate_vllm.py --model /path/to/model --prompt "Hello" --max_tokens 32
"""

from __future__ import annotations

import argparse
import sys
import time

from vllm import LLM, SamplingParams


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM smoke-test")
    parser.add_argument("--model", type=str, required=True, help="Path or name of the model")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                        help="Prompt to generate from")
    parser.add_argument("--max_tokens", type=int, default=32, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    print("Loading vLLM model: ", args.model)
    # Simple health checks
    try:
        import torch
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    llm = LLM(model=args.model, enable_prefix_caching=False)
    tokenizer = llm.get_tokenizer()

    prompt_tokens = tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a helpful assistant.\nthinking on\n"},
        {"role": "user", "content": args.prompt}], add_generation_prompt=True)

    #breakpoint()
    
    print("Message: ", tokenizer.decode(prompt_tokens))

    #print("Tokenizer OK. Encoding prompt...")
    #prompt_tokens = tokenizer.encode(prompt_tokens)
    print(f"Prompt token count: {len(prompt_tokens)}")

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    print("Generating...")
    t0 = time.time()
    outputs = llm.generate([prompt_tokens], sampling_params)
    dt = time.time() - t0

    out = outputs[0].outputs[0]
    text = out.text
    token_ids = getattr(out, "token_ids", None)

    print(f"Generated in {dt:.2f}s")
    print("---- OUTPUT ----")
    print(text)
    print("----------------")
    if token_ids is not None:
        print(f"Generated token ids (count={len(token_ids)}): {token_ids[:20]}{'...' if len(token_ids)>20 else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
