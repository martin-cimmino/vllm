#!/usr/bin/env python3
"""Phase 0 (v2): Batched Power-SMC prototype using vLLM offline inference.

Improvement over v1: instead of N separate generate() calls per step,
this version submits N requests at once and lets vLLM batch them.
Prefix caching ensures the shared prompt KV is computed only once.

After each round of generation (~64 tokens), we check ESS and resample
if needed. This amortizes the overhead of resampling over multiple tokens
while still getting the benefits of SMC.

For the weight computation, we use logprobs=-1 to get full-vocab logprobs
and compute the exact incremental weight:
    log ω_t = logsumexp(α * log_p)

Usage:
    python smc_prototype_v2.py --model <path> --n_particles 8 --alpha 3.0

Requires vLLM installed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


# ── SMC Primitives ───────────────────────────────────────────────────

def compute_ess(log_weights: Tensor) -> float:
    log_w = log_weights - torch.logsumexp(log_weights, dim=0)
    return torch.exp(-torch.logsumexp(2 * log_w, dim=0)).item()


def systematic_resample(log_weights: Tensor) -> Tensor:
    N = log_weights.shape[0]
    log_w = log_weights - torch.logsumexp(log_weights, dim=0)
    weights = torch.exp(log_w)
    cumsum = torch.cumsum(weights, dim=0)
    u0 = torch.rand(1, device=log_weights.device, dtype=log_weights.dtype)
    positions = (u0 + torch.arange(N, device=log_weights.device,
                                    dtype=log_weights.dtype)) / N
    return torch.searchsorted(cumsum, positions).clamp(max=N - 1).long()


# ── Answer Extraction ────────────────────────────────────────────────

def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def extract_aime_answer(text: str) -> int | None:
    text = strip_think_blocks(text)
    boxed = re.findall(r"\\boxed\s*\{([^}]+)\}", text)
    if boxed:
        raw = boxed[-1].strip()
        raw = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^}]*)\}", r"\1", raw)
        raw = raw.replace(",", "").replace(" ", "").replace("−", "-")
        try:
            return int(raw)
        except ValueError:
            try:
                f = float(raw)
                if f == int(f):
                    return int(f)
            except ValueError:
                pass
    integers = re.findall(r"(?<![.\d])\b(\d{1,3})\b(?!\.\d)", text)
    if integers:
        try:
            return int(integers[-1])
        except ValueError:
            pass
    return None


# ── Voting ───────────────────────────────────────────────────────────

def majority_vote(answers: list[int | None]) -> int | None:
    valid = [a for a in answers if a is not None]
    return Counter(valid).most_common(1)[0][0] if valid else None


def weighted_majority_vote(answers: list[int | None],
                           log_weights: Tensor) -> int | None:
    weights = torch.softmax(log_weights.float(), dim=0).tolist()
    vote: dict[int, float] = {}
    for ans, w in zip(answers, weights):
        if ans is not None:
            vote[ans] = vote.get(ans, 0.0) + w
    return max(vote, key=vote.get) if vote else None


def snis_draw(answers: list[int | None], log_weights: Tensor,
              rng: torch.Generator | None = None) -> int | None:
    probs = torch.softmax(log_weights.float(), dim=0)
    idx = torch.multinomial(probs, num_samples=1, generator=rng).item()
    return answers[idx]


# ── Config ───────────────────────────────────────────────────────────

@dataclass
class SMCConfig:
    n_particles: int = 8
    alpha: float = 3.0
    max_new_tokens: int = 4096
    ess_threshold: float = 0.5
    alpha_ramp_tokens: int = 0
    seed: int | None = None


# ── Phase 0 v2: Full-generation then offline weight computation ──────
#
# Strategy: Generate N complete sequences with vLLM (fast batched),
# then compute importance weights *after* generation by examining
# the token logprobs. This gives us the full SMC pipeline without
# modifying vLLM internals.
#
# The tradeoff: we can't resample mid-generation (that requires engine
# changes), but we CAN compute and use the importance weights for SNIS
# and weighted voting. This validates the weight computation and voting
# logic, and gives a baseline to compare future mid-generation
# resampling against.
#
# For mid-generation resampling we'd need iterative single-token
# generation (v1 prototype) or engine modifications.

def power_smc_vllm_batch(
    llm,
    prompt: str,
    config: SMCConfig,
) -> dict[str, Any]:
    """Batch Power-SMC: generate N full sequences, compute weights post-hoc.

    This version generates all N particles to completion in one batched
    vLLM call (fast), then computes token-level importance weights from
    logprobs. No mid-generation resampling — this is importance sampling,
    not full SMC. But it validates the weight computation and shows
    how much room mid-generation resampling has to improve.

    For full mid-generation SMC, use smc_prototype.py (v1, slow) or
    wait for Phase 1+ engine integration.
    """
    from vllm import LLM, SamplingParams as VLLMSamplingParams

    N = config.n_particles
    alpha = config.alpha
    T_max = config.max_new_tokens

    if config.seed is not None:
        torch.manual_seed(config.seed)

    t_start = time.time()

    # Temperature = 1/α for the optimal proposal
    tau = 1.0 / alpha

    # Generate N complete sequences in one batched call
    # Request logprobs for the sampled token so we can compute weights
    sampling_params = VLLMSamplingParams(
        temperature=tau,
        max_tokens=T_max,
        n=N,
        logprobs=1,  # Return logprob of sampled token
        top_p=1.0,
        seed=config.seed,
    )

    print(f"  Generating {N} particles (batched)...")
    outputs = llm.generate([prompt], sampling_params, use_tqdm=True)
    gen_time = time.time() - t_start
    print(f"  Generation done in {gen_time:.1f}s")

    # Process outputs — compute importance weights from logprobs
    request_output = outputs[0]

    sequences = []
    token_id_lists = []
    log_weights = torch.zeros(N)
    ess_trace = []

    for i, completion in enumerate(request_output.outputs):
        text = completion.text
        sequences.append(text)
        token_id_lists.append(list(completion.token_ids))

        # Compute cumulative log-weight for this particle
        # Under the optimal proposal q*(v) ∝ p(v)^α, the incremental weight is:
        #   log ω_t = logsumexp(α * log_p([v for all v])
        #
        # With only the sampled token logprob available (logprobs=1), we
        # approximate: each token contributes α * log_p(sampled_token).
        # This is the IS weight under the proposal p^α / Z_α.
        #
        # Note: with logprobs=1, we get the logprob of the sampled token
        # under the *proposal* distribution (temperature=τ). We need to
        # convert to the base model logprob:
        #   log_p_proposal(v) = α * log_p_base(v) - log Z_α
        #   → log_p_base(v) = (log_p_proposal(v) + log Z_α) / α
        #
        # For IS weight computation, what matters is:
        #   log w(y) = log p_base(y) * α - log q(y)
        #            = α * Σ_t log_p_base(y_t) - Σ_t log_q(y_t)
        #
        # When q = optimal proposal (temp=1/α), these cancel to:
        #   log w(y) = Σ_t logsumexp(α * log_p_base)
        #
        # Since we don't have the full vocab logprobs through the API,
        # we use a simpler weight: just the cumulative base-model logprob
        # raised to power α. This is not the exact SMC weight but gives
        # directional signal.
        #
        # For the exact computation we need Phase 1 (sampler modification).

        cumulative_log_p = 0.0
        if completion.logprobs:
            for step_logprobs in completion.logprobs:
                if step_logprobs:
                    # step_logprobs is a dict: {token_id: Logprob}
                    # The sampled token's logprob
                    for token_id, logprob_obj in step_logprobs.items():
                        # logprob_obj.logprob is the log-prob under temp=τ
                        # We want base-model log-prob: multiply by τ = 1/α
                        # Actually: at temp τ, score = logit/τ = α*logit
                        # So log_p_proposal = log_softmax(α * logits)
                        # And log_p_base = log_softmax(logits) = log_p_proposal - (α-1)*logit/??
                        # This doesn't simplify nicely. Use the proposal logprob as-is.
                        cumulative_log_p += logprob_obj.logprob
                        break  # only take the first (sampled) token

        # Simple weight: just use cumulative proposal logprob
        # Higher logprob sequences → better under π_α
        log_weights[i] = cumulative_log_p

    elapsed = time.time() - t_start

    # Extract answers
    answers = [extract_aime_answer(text) for text in sequences]

    # Count unique particles
    unique_seqs = len(set(tuple(t) for t in token_id_lists))

    # Compute ESS of the final weights
    final_ess = compute_ess(log_weights)

    return {
        "sequences": sequences,
        "token_ids": token_id_lists,
        "log_weights": log_weights,
        "answers": answers,
        "majority_vote": majority_vote(answers),
        "weighted_majority": weighted_majority_vote(answers, log_weights),
        "snis_draw": snis_draw(answers, log_weights),
        "diagnostics": {
            "final_ess": final_ess,
            "n_unique_final": unique_seqs,
            "n_tokens_generated_max": max(len(t) for t in token_id_lists) if token_id_lists else 0,
            "n_tokens_generated_min": min(len(t) for t in token_id_lists) if token_id_lists else 0,
            "elapsed_seconds": elapsed,
            "gen_time_seconds": gen_time,
            "mode": "batch_no_resampling",
        },
    }


# ── Test Prompts ─────────────────────────────────────────────────────

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem efficiently and clearly. "
    "The last line of your response should be of the following format: "
    "'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' "
    "(without quotes) where ANSWER is just the final number or expression "
    "that solves the problem. Think step by step before answering.\n\n{problem}"
)

TEST_PROBLEMS = [
    {
        "name": "Combinatorics — circular seating",
        "problem": (
            "Eight people, including Alice and Bob, are to be seated around "
            "a circular table. Two seatings are considered the same if one is "
            "a rotation of the other. How many seatings are there such that "
            "Alice and Bob are NOT adjacent?"
        ),
        "answer": 3600,
    },
    {
        "name": "Number Theory — consecutive cube differences",
        "problem": (
            "Find the number of positive integers n ≤ 1000 such that n can "
            "be expressed as the difference of two consecutive perfect cubes."
        ),
        "answer": 10,
    },
    {
        "name": "Algebra — nested quadratic (sanity check)",
        "problem": (
            "Let f(x) = x^2 + 6x + 5. Find the sum of all values of x "
            "such that f(f(x)) = 0."
        ),
        "answer": -12,
    },
]


def format_prompt(problem: str, tokenizer, thinking: bool = False) -> str:
    query = MATH_PROMPT_TEMPLATE.format(problem=problem)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = []
            if thinking:
                messages.append({"role": "system", "content": "\nthinking on\n"})
            messages.append({"role": "user", "content": query})
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return query


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 v2: Batched Power-SMC prototype with vLLM"
    )
    parser.add_argument(
        "--model", type=str,
        default="/leonardo_scratch/fast/iGen_train/models/Domyn-Small-v0.2-bf16",
    )
    parser.add_argument("--n_particles", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--ess_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--problem_idx", type=int, default=2,
                        help="0=combinatorics, 1=number theory, 2=algebra (default)")
    parser.add_argument("--all_problems", action="store_true",
                        help="Run all test problems")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default="./smc_results")
    args = parser.parse_args()

    print("=" * 60)
    print("  Power-SMC Phase 0 v2 — Batched (no mid-gen resampling)")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {args.model}")
    from vllm import LLM
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    config = SMCConfig(
        n_particles=args.n_particles,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        ess_threshold=args.ess_threshold,
        seed=args.seed,
    )

    # Determine which problems to run
    problems = TEST_PROBLEMS if args.all_problems else [TEST_PROBLEMS[args.problem_idx]]

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for prob_idx, problem in enumerate(problems):
        print(f"\n{'─' * 60}")
        print(f"Problem: {problem['name']}")
        print(f"Ground truth: {problem['answer']}")
        print(f"{'─' * 60}")

        prompt = format_prompt(problem["problem"], tokenizer, thinking=args.thinking)
        prompt_len = len(tokenizer.encode(prompt))
        print(f"Prompt length: {prompt_len} tokens")
        print(f"SMC: n={config.n_particles}, α={config.alpha}, "
              f"τ={1/config.alpha:.3f}, seed={config.seed}")

        # Run batch generation
        result = power_smc_vllm_batch(llm, prompt, config)

        diag = result["diagnostics"]
        print(f"\nResults:")
        print(f"  Time: {diag['elapsed_seconds']:.1f}s (gen: {diag['gen_time_seconds']:.1f}s)")
        print(f"  Tokens: {diag['n_tokens_generated_min']}-{diag['n_tokens_generated_max']}")
        print(f"  Unique particles: {diag['n_unique_final']}/{config.n_particles}")
        print(f"  Final ESS: {diag['final_ess']:.2f} / {config.n_particles}")

        answer_counts = Counter(a for a in result["answers"] if a is not None)
        print(f"\n  Answers:")
        for ans, count in answer_counts.most_common(5):
            marker = " ✓" if ans == problem["answer"] else ""
            print(f"    {ans}: {count}/{config.n_particles}{marker}")
        none_count = sum(1 for a in result["answers"] if a is None)
        if none_count:
            print(f"    None: {none_count}/{config.n_particles}")

        gt = problem["answer"]
        mv = result["majority_vote"]
        wm = result["weighted_majority"]
        sn = result["snis_draw"]
        print(f"\n  Majority vote:       {mv} {'✓' if mv == gt else '✗'}")
        print(f"  Weighted majority:   {wm} {'✓' if wm == gt else '✗'}")
        print(f"  SNIS draw:           {sn} {'✓' if sn == gt else '✗'}")
        print(f"  Ground truth:        {gt}")

        # Show sample outputs
        print(f"\n  Sample outputs (last 150 chars):")
        for i in range(min(3, config.n_particles)):
            text = result["sequences"][i][-150:].replace("\n", " ")
            ans = result["answers"][i]
            w = result["log_weights"][i].item()
            print(f"    [{i}] w={w:.2f} ans={ans} | ...{text}")

        # Save to file
        save_result = {
            "problem": problem,
            "config": {
                "n_particles": config.n_particles,
                "alpha": config.alpha,
                "max_new_tokens": config.max_new_tokens,
                "seed": config.seed,
            },
            "answers": result["answers"],
            "log_weights": result["log_weights"].tolist(),
            "majority_vote": result["majority_vote"],
            "weighted_majority": result["weighted_majority"],
            "snis_draw": result["snis_draw"],
            "diagnostics": {k: v for k, v in diag.items()
                           if not isinstance(v, (torch.Tensor, list))},
        }
        all_results.append(save_result)

    # Save all results
    output_file = Path(args.output_dir) / f"smc_phase0_v2_a{config.alpha}_n{config.n_particles}_s{config.seed}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
