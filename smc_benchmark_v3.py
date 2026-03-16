#!/usr/bin/env python3
"""Phase 1–3 Validation Benchmark: compare in-GPU exact SMC weights
against Phase 0 post-hoc logprob approximation.

Note that particle resampling is not performed in this benchmark — all particles are allowed to run to completion.

Captures per-step weights via monkey-patch on SMCController.accumulate()
(same pattern as test_smc_accumulate_called_during_inference). The hook
receives smc_log_weights: dict[str, float] each step even though no
groups are registered.

Usage:
    python smc_benchmark_v3.py --model /leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16 --n_particles 16 --alpha 2.0 --problem_idx 2 --max_new_tokens 4096
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
from pathlib import Path
from typing import Any

import torch

# Force single-process mode so monkey-patch works in-process
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


# ── SMC Primitives ───────────────────────────────────────────────────

def compute_ess(log_weights: torch.Tensor) -> float:
    log_w = log_weights - torch.logsumexp(log_weights, dim=0)
    return torch.exp(-torch.logsumexp(2 * log_w, dim=0)).item()


def systematic_resample(log_weights: torch.Tensor) -> torch.Tensor:
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
                           log_weights: torch.Tensor) -> int | None:
    weights = torch.softmax(log_weights.float(), dim=0).tolist()
    vote: dict[int, float] = {}
    for ans, w in zip(answers, weights):
        if ans is not None:
            vote[ans] = vote.get(ans, 0.0) + w
    return max(vote, key=vote.get) if vote else None


def snis_draw(answers: list[int | None], log_weights: torch.Tensor,
              rng: torch.Generator | None = None) -> int | None:
    probs = torch.softmax(log_weights.float(), dim=0)
    idx = torch.multinomial(probs, num_samples=1, generator=rng).item()
    return answers[idx]


# ── Weight Capture ───────────────────────────────────────────────────

class WeightCapture:
    """Context manager that captures per-step SMC weights from the engine.

    Monkey-patches SMCController.accumulate() to record the
    smc_log_weights dict passed each step.
    """

    def __init__(self):
        self.step_weights: list[dict[str, float]] = []
        self._original = None

    def __enter__(self):
        from vllm.v1.engine.smc_controller import SMCController

        self._original = SMCController.accumulate

        capture = self

        def capturing_accumulate(ctrl_self, smc_log_weights):
            capture.step_weights.append(dict(smc_log_weights))
            return capture._original(ctrl_self, smc_log_weights)

        SMCController.accumulate = capturing_accumulate
        return self

    def __exit__(self, *exc):
        from vllm.v1.engine.smc_controller import SMCController

        SMCController.accumulate = self._original

    def cumulative_weights(self) -> dict[str, float]:
        """Sum per-step weights into cumulative per-request totals."""
        totals: dict[str, float] = {}
        for step in self.step_weights:
            for req_id, w in step.items():
                totals[req_id] = totals.get(req_id, 0.0) + w
        return totals

    def step_count_per_request(self) -> dict[str, int]:
        """Count how many steps each request appeared in."""
        counts: dict[str, int] = {}
        for step in self.step_weights:
            for req_id in step:
                counts[req_id] = counts.get(req_id, 0) + 1
        return counts


# ── Request ID → particle index mapping ──────────────────────────────

def parse_particle_index(req_id: str) -> int | None:
    """Extract particle index from '<idx>_<parent_id>' naming convention."""
    parts = req_id.split("_", 1)
    if len(parts) == 2:
        try:
            return int(parts[0])
        except ValueError:
            pass
    return None


def map_request_ids_to_particles(
    req_ids: list[str], n_particles: int
) -> dict[str, int]:
    """Map request IDs to particle indices [0, N)."""
    mapping: dict[str, int] = {}

    # Try the '<idx>_<parent_id>' convention first
    for req_id in req_ids:
        idx = parse_particle_index(req_id)
        if idx is not None and 0 <= idx < n_particles:
            mapping[req_id] = idx

    # If convention didn't work, fall back to sorted order
    if len(mapping) != n_particles:
        mapping = {}
        for i, req_id in enumerate(sorted(req_ids)):
            if i < n_particles:
                mapping[req_id] = i

    return mapping


# ── Rank correlation ─────────────────────────────────────────────────

def spearman_rank_correlation(a: list[float], b: list[float]) -> float:
    """Compute Spearman rank correlation between two lists."""
    n = len(a)
    if n < 2:
        return float("nan")

    def rank(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for r, i in enumerate(order):
            ranks[i] = float(r)
        return ranks

    ra, rb = rank(a), rank(b)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


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


# ── Main benchmark ───────────────────────────────────────────────────

def run_benchmark(
    llm,
    prompt: str,
    n_particles: int,
    alpha: float,
    max_tokens: int,
    seed: int | None,
) -> dict[str, Any]:
    """Run one problem: generate with weight capture, compare engine vs logprob weights."""
    from vllm import SamplingParams

    sp = SamplingParams(
        n=n_particles,
        smc_alpha=alpha,       # triggers in-GPU weight computation; auto-sets temperature=1/α
        max_tokens=max_tokens,
        logprobs=1,            # for Phase 0-style comparison
        top_p=1.0,
        seed=seed,
    )

    capture = WeightCapture()
    t_start = time.time()

    print(f"  Generating {n_particles} particles (smc_alpha={alpha})...")
    with capture:
        outputs = llm.generate([prompt], sp, use_tqdm=True)
    gen_time = time.time() - t_start
    print(f"  Generation done in {gen_time:.1f}s")
    print(f"  Captured {len(capture.step_weights)} engine steps")

    request_output = outputs[0]

    # ── Engine weights (exact, from accumulate hook) ──
    cum_weights = capture.cumulative_weights()
    step_counts = capture.step_count_per_request()

    # Map request IDs to particle indices
    req_ids = list(cum_weights.keys())
    id_map = map_request_ids_to_particles(req_ids, n_particles)

    engine_weights = torch.zeros(n_particles)
    engine_step_counts = [0] * n_particles
    for req_id, w in cum_weights.items():
        if req_id in id_map:
            idx = id_map[req_id]
            engine_weights[idx] = w
            engine_step_counts[idx] = step_counts.get(req_id, 0)

    # ── Logprob weights (approximate Phase 0 style) ──
    logprob_weights = torch.zeros(n_particles)
    sequences = []
    answers_list = []
    token_counts = []

    for i, completion in enumerate(request_output.outputs):
        sequences.append(completion.text)
        token_counts.append(len(completion.token_ids))
        answers_list.append(extract_aime_answer(completion.text))

        cumulative_log_p = 0.0
        if completion.logprobs:
            for step_logprobs in completion.logprobs:
                if step_logprobs:
                    for token_id, logprob_obj in step_logprobs.items():
                        cumulative_log_p += logprob_obj.logprob
                        break  # only the sampled token
        logprob_weights[i] = cumulative_log_p

    # ── Diagnostics ──
    engine_ess = compute_ess(engine_weights)
    logprob_ess = compute_ess(logprob_weights)

    engine_finite = all(math.isfinite(w) for w in engine_weights.tolist())
    engine_nonpos = all(w <= 1e-9 for w in engine_weights.tolist())

    rank_corr = spearman_rank_correlation(
        engine_weights.tolist(), logprob_weights.tolist()
    )

    # ── Voting ──
    mv = majority_vote(answers_list)
    wv_engine = weighted_majority_vote(answers_list, engine_weights)
    wv_logprob = weighted_majority_vote(answers_list, logprob_weights)
    snis_engine = snis_draw(answers_list, engine_weights)
    snis_logprob = snis_draw(answers_list, logprob_weights)

    unique_seqs = len(set(tuple(completion.token_ids)
                         for completion in request_output.outputs))

    return {
        "sequences": sequences,
        "answers": answers_list,
        "engine_weights": engine_weights.tolist(),
        "logprob_weights": logprob_weights.tolist(),
        "engine_ess": engine_ess,
        "logprob_ess": logprob_ess,
        "step_count": len(capture.step_weights),
        "engine_step_counts_per_particle": engine_step_counts,
        "token_counts": token_counts,
        "rank_correlation": rank_corr,
        "majority_vote": mv,
        "weighted_vote_engine": wv_engine,
        "weighted_vote_logprob": wv_logprob,
        "snis_engine": snis_engine,
        "snis_logprob": snis_logprob,
        "diagnostics": {
            "engine_weights_all_finite": engine_finite,
            "engine_weights_all_nonpos": engine_nonpos,
            "n_unique_particles": unique_seqs,
            "n_captured_request_ids": len(cum_weights),
            "elapsed_seconds": time.time() - t_start,
            "gen_time_seconds": gen_time,
        },
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1-3 Validation: exact in-GPU vs post-hoc logprob weights"
    )
    parser.add_argument(
        "--model", type=str,
        default="/leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16",
    )
    parser.add_argument("--n_particles", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
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

    print("=" * 70)
    print("  Power-SMC Phase 1-3 Validation Benchmark")
    print("  Exact in-GPU weights vs Phase 0 post-hoc logprob approximation")
    print("=" * 70)

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

    problems = TEST_PROBLEMS if args.all_problems else [TEST_PROBLEMS[args.problem_idx]]

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for prob_idx, problem in enumerate(problems):
        print(f"\n{'─' * 70}")
        print(f"Problem: {problem['name']}")
        print(f"Ground truth: {problem['answer']}")
        print(f"{'─' * 70}")

        prompt = format_prompt(problem["problem"], tokenizer, thinking=args.thinking)
        prompt_len = len(tokenizer.encode(prompt))
        print(f"Prompt length: {prompt_len} tokens")
        print(f"SMC: n={args.n_particles}, α={args.alpha}, "
              f"τ={1/args.alpha:.3f}, seed={args.seed}")

        result = run_benchmark(
            llm, prompt,
            n_particles=args.n_particles,
            alpha=args.alpha,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        )

        gt = problem["answer"]
        diag = result["diagnostics"]

        # ── Print comparison ──
        print(f"\n  {'─' * 50}")
        print(f"  Weight comparison ({args.n_particles} particles):")
        print(f"  {'─' * 50}")

        ew = result["engine_weights"]
        lw = result["logprob_weights"]
        print(f"  Engine weights (exact):    [{', '.join(f'{w:.4f}' for w in ew)}]")
        print(f"  Logprob weights (approx):  [{', '.join(f'{w:.2f}' for w in lw)}]")
        print(f"  Engine ESS:   {result['engine_ess']:.2f} / {args.n_particles}")
        print(f"  Logprob ESS:  {result['logprob_ess']:.2f} / {args.n_particles}")
        print(f"  Rank correlation: {result['rank_correlation']:.4f}")
        print(f"  Steps captured: {result['step_count']}")
        print(f"  Engine weights finite: {diag['engine_weights_all_finite']}")
        print(f"  Engine weights ≤ 0:    {diag['engine_weights_all_nonpos']}")

        # Per-particle breakdown
        print(f"\n  Per-particle breakdown:")
        print(f"  {'Idx':>3}  {'Engine w':>10}  {'Logprob w':>10}  {'Steps':>5}  {'Tokens':>6}  {'Answer':>8}")
        for i in range(args.n_particles):
            ans = result["answers"][i]
            ans_str = str(ans) if ans is not None else "None"
            marker = " ✓" if ans == gt else ""
            print(f"  {i:>3}  {ew[i]:>10.4f}  {lw[i]:>10.2f}  "
                  f"{result['engine_step_counts_per_particle'][i]:>5}  "
                  f"{result['token_counts'][i]:>6}  {ans_str:>8}{marker}")

        # Voting results
        mv = result["majority_vote"]
        wve = result["weighted_vote_engine"]
        wvl = result["weighted_vote_logprob"]
        sne = result["snis_engine"]
        snl = result["snis_logprob"]
        print(f"\n  Voting:")
        print(f"    Majority vote:          {mv} {'✓' if mv == gt else '✗'}")
        print(f"    Weighted vote (engine):  {wve} {'✓' if wve == gt else '✗'}")
        print(f"    Weighted vote (logprob): {wvl} {'✓' if wvl == gt else '✗'}")
        print(f"    SNIS (engine):           {sne} {'✓' if sne == gt else '✗'}")
        print(f"    SNIS (logprob):          {snl} {'✓' if snl == gt else '✗'}")
        print(f"    Ground truth:            {gt}")

        # Answer distribution
        answer_counts = Counter(a for a in result["answers"] if a is not None)
        print(f"\n  Answer distribution:")
        for ans, count in answer_counts.most_common(5):
            marker = " ✓" if ans == gt else ""
            print(f"    {ans}: {count}/{args.n_particles}{marker}")
        none_count = sum(1 for a in result["answers"] if a is None)
        if none_count:
            print(f"    None: {none_count}/{args.n_particles}")

        print(f"\n  Time: {diag['elapsed_seconds']:.1f}s "
              f"(gen: {diag['gen_time_seconds']:.1f}s)")

        # Save result (exclude full sequences for JSON)
        save_result = {
            "problem": problem,
            "config": {
                "n_particles": args.n_particles,
                "alpha": args.alpha,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            },
            "engine_weights": result["engine_weights"],
            "logprob_weights": result["logprob_weights"],
            "engine_ess": result["engine_ess"],
            "logprob_ess": result["logprob_ess"],
            "step_count": result["step_count"],
            "engine_step_counts_per_particle": result["engine_step_counts_per_particle"],
            "token_counts": result["token_counts"],
            "rank_correlation": result["rank_correlation"],
            "answers": result["answers"],
            "majority_vote": result["majority_vote"],
            "weighted_vote_engine": result["weighted_vote_engine"],
            "weighted_vote_logprob": result["weighted_vote_logprob"],
            "snis_engine": result["snis_engine"],
            "snis_logprob": result["snis_logprob"],
            "diagnostics": diag,
        }
        all_results.append(save_result)

    # Save all results
    output_file = (
        Path(args.output_dir)
        / f"smc_benchmark_v3_a{args.alpha}_n{args.n_particles}_s{args.seed}.json"
    )
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
