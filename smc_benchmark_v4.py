#!/usr/bin/env python3
"""Phase 2 Resampling Benchmark: live mid-generation resampling vs baseline.

Runs each problem twice:
  - Baseline:   smc_ess_threshold very small → no resampling fires
  - Resampling: smc_ess_threshold=tau        → resampling fires when ESS < tau

Captures per-step ESS and resampling events via monkey-patches on
SMCController.accumulate() and SMCController.maybe_resample().

Key metrics reported:
  - ESS trajectory: min / mean / final ESS during generation
  - Resampling events: how often fired, which steps
  - Answer accuracy: majority vote, weighted vote with accumulated weights
  - Diversity: number of unique final answers
  - Wall-clock time: generation overhead from resampling

Usage:
    python smc_benchmark_v4.py \\
        --model /leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16 \\
        --n_particles 8 --alpha 2.0 --ess_threshold 0.5 \\
        --problem_idx 2 --max_new_tokens 2048

    python smc_benchmark_v4.py --n_particles 16 --alpha 4.0 --ess_threshold 0.5 --problem_idx 2 --max_new_tokens 4096 --alpha_ramp_tokens 100 --thinking
    # force resampling by increasing threshold:
    python smc_benchmark_v4.py --n_particles 16 --alpha 5.0 --ess_threshold 0.5 --problem_idx 4 --max_new_tokens 8192 --alpha_ramp_tokens 1000 --thinking

    python smc_benchmark_v4.py --n_particles 16 --alpha 5.0 --ess_threshold 0.9 --problem_idx 4 --max_new_tokens 4096 --alpha_ramp_tokens 3000 --thinking

    # Run all problems, compare across thresholds:
    python smc_benchmark_v4.py --all_problems --n_particles 8 --alpha 2.0 \\
        --ess_threshold 0.5 --compare_baseline
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import random

import torch

# Force single-process mode so monkey-patches work in-process
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


# ── SMC Primitives ────────────────────────────────────────────────────

def compute_ess(log_weights: list[float]) -> float:
    """Normalized ESS ∈ [0, 1] from a list of log-weights."""
    n = len(log_weights)
    if n == 0:
        return 0.0
    t = torch.tensor(log_weights, dtype=torch.float64)
    log_w = t - torch.logsumexp(t, dim=0)
    return torch.exp(-torch.logsumexp(2 * log_w, dim=0)).item()


# ── Answer Extraction ─────────────────────────────────────────────────

def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_boxed_content(text: str) -> list[str]:
    """Find all \\boxed{...} contents, correctly handling nested braces."""
    results = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start:i-1])
    return results


def _clean_boxed(raw: str) -> str:
    # \frac{a}{b} cannot be reliably parsed as an integer (stripping \frac and
    # braces would concatenate numerator and denominator digits). AIME answers
    # are always integers, so bail out and let later stages handle it.
    if re.search(r"\\frac\s*\{", raw):
        return ""
    raw = re.sub(r"\\(?:text|mathrm|mathbf|mbox)\s*\{([^}]*)\}", r"\1", raw)
    raw = re.sub(r"\\(?:left|right|big|Big)\s*[()[\]|]", "", raw)
    raw = re.sub(r"\\[a-zA-Z]+", "", raw)
    raw = re.sub(r"[{}\[\]]", "", raw)
    raw = raw.replace(",", "").replace(" ", "").replace("−", "-")
    return raw.strip()


def extract_aime_answer(text: str) -> int | None:
    text = strip_think_blocks(text)

    # Stage 1: nested-brace \boxed{} extraction
    boxed = _extract_boxed_content(text)
    if boxed:
        raw = _clean_boxed(boxed[-1])
        try:
            return int(raw)
        except ValueError:
            try:
                f = float(raw)
                if f == int(f):
                    return int(f)
            except ValueError:
                pass
        # \boxed{} found but content not parseable as integer — the model's
        # declared answer is unparseable, so don't fall through to digit fallbacks
        # (which would grab spurious digits from inside the boxed expression).
        return None

    # Stage 2: "= N" at tail / "answer is N"
    tail = text[-200:]
    for pat in [r"=\s*(-?\d+)\s*[.$\n]?", r"(?:answer\s*(?:is|:)\s*)(-?\d+)"]:
        m = re.findall(pat, tail, re.IGNORECASE)
        if m:
            try:
                return int(m[-1])
            except ValueError:
                pass

    # Stage 3: last 1-3 digit standalone integer in final 300 chars
    tail = text[-300:]
    integers = re.findall(r"(?<![.\d])\b(\d{1,3})\b(?!\.\d)", tail)
    if integers:
        try:
            return int(integers[-1])
        except ValueError:
            pass

    return None


# ── Voting ────────────────────────────────────────────────────────────

def majority_vote(answers: list[int | None]) -> int | None:
    valid = [a for a in answers if a is not None]
    return Counter(valid).most_common(1)[0][0] if valid else None


def weighted_majority_vote(answers: list[int | None],
                           log_weights: list[float]) -> int | None:
    t = torch.tensor(log_weights, dtype=torch.float32)
    weights = torch.softmax(t, dim=0).tolist()
    vote: dict[int, float] = {}
    for ans, w in zip(answers, weights):
        if ans is not None:
            vote[ans] = vote.get(ans, 0.0) + w
    return max(vote, key=vote.__getitem__) if vote else None  # type: ignore[arg-type]

def snis_draw(answers: list[int | None], log_weights: list[float], rng: torch.Generator | None = None) -> int | None:
    t = torch.tensor(log_weights, dtype=torch.float32)
    probs = torch.softmax(t, dim=0)
    idx = torch.multinomial(probs, num_samples=1, generator=rng).item()
    return answers[idx] if 0 <= idx < len(answers) else None


# ── Instrumentation ───────────────────────────────────────────────────

class SMCInstrumentation:
    """Context manager that instruments SMCController for observation.

    Patches both `accumulate()` (to record per-step weights and ESS)
    and `maybe_resample()` (to record resampling events and actions).

    Thread-safe only for single-process mode (VLLM_ENABLE_V1_MULTIPROCESSING=0).
    """

    def __init__(self):
        # Per-step records: list of {step: int, ess: float, req_weights: dict}
        self.steps: list[dict[str, Any]] = []
        # Resample events: list of {step: int, parent_id: str, loser_ids, new_ids}
        self.resample_events: list[dict[str, Any]] = []
        # Cumulative per-ID weights (for any ID seen, including post-resample)
        self._cum_weights: dict[str, float] = {}
        # ID remapping chain: new_id → original_child_id
        self._id_to_original: dict[str, str] = {}
        # Reference to the live SMCController instance (captured on first accumulate)
        self._controller: object = None
        self._orig_accumulate = None
        self._orig_maybe_resample = None

    def __enter__(self):
        from vllm.v1.engine.smc_controller import SMCController
        self._orig_accumulate = SMCController.accumulate
        self._orig_maybe_resample = SMCController.maybe_resample
        instr = self

        def patched_accumulate(ctrl_self, smc_log_weights: dict[str, float]) -> None:
            step_idx = len(instr.steps)
            instr._controller = ctrl_self  # capture on first call
            for req_id, w in smc_log_weights.items():
                instr._cum_weights[req_id] = instr._cum_weights.get(req_id, 0.0) + w
            # Call the original first so group.log_weights are already updated,
            # then compute ESS from the post-update weights.
            result = instr._orig_accumulate(ctrl_self, smc_log_weights)
            group_ess: dict[str, float] = {}
            for pid, group in ctrl_self._groups.items():
                active_weights = [
                    lw for i, lw in enumerate(group.log_weights) if i not in group.frozen_weights
                ]
                #ess = ctrl_self.compute_ess(active_weights)
                ess = ctrl_self.compute_ess(group.log_weights)
                group_ess[pid] = ess
            instr.steps.append({
                "step": step_idx,
                "req_weights": dict(smc_log_weights),
                "group_ess": group_ess,
            })
            return result

        def patched_maybe_resample(ctrl_self, requests, token_snapshots=None) -> dict:
            step_idx = len(instr.steps) - 1  # step just accumulated
            actions = instr._orig_maybe_resample(ctrl_self, requests, token_snapshots)
            for pid, action in actions.items():
                event = {
                    "step": step_idx,
                    "parent_id": pid,
                    "loser_ids": list(action.loser_request_ids),
                    "new_particles": [
                        {
                            "new_id": p.new_request_id,
                            "ancestor_id": p.ancestor_request_id,
                            "slot": p.slot_index,
                            "num_output_tokens": p.num_output_tokens,
                            "remaining_tokens": p.original_max_tokens - p.num_output_tokens,
                        }
                        for p in action.new_particles
                    ],
                }
                instr.resample_events.append(event)
                # Mirror the controller's _id_to_original into our local copy
                instr._id_to_original.update(ctrl_self._id_to_original)
            return actions

        SMCController.accumulate = patched_accumulate
        SMCController.maybe_resample = patched_maybe_resample
        return self

    def __exit__(self, *exc):
        from vllm.v1.engine.smc_controller import SMCController
        SMCController.accumulate = self._orig_accumulate
        SMCController.maybe_resample = self._orig_maybe_resample

    # ── Derived metrics ──────────────────────────────────────────────

    def ess_trajectory(self) -> list[float]:
        """ESS per step (mean across groups if multiple)."""
        result = []
        for s in self.steps:
            ge = s["group_ess"]
            if ge:
                result.append(sum(ge.values()) / len(ge))
        return result

    def min_ess(self) -> float:
        traj = self.ess_trajectory()
        return min(traj) if traj else float("nan")

    def mean_ess(self) -> float:
        traj = self.ess_trajectory()
        return sum(traj) / len(traj) if traj else float("nan")

    def final_ess(self) -> float:
        traj = self.ess_trajectory()
        return traj[-1] if traj else float("nan")

    def cumulative_weights_by_original_id(self) -> dict[str, float]:
        """Sum cumulative weights, resolving all IDs to original child IDs."""
        totals: dict[str, float] = {}
        for req_id, w in self._cum_weights.items():
            original = self._id_to_original.get(req_id, req_id)
            totals[original] = totals.get(original, 0.0) + w
        return totals


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


# ── Test Prompts ──────────────────────────────────────────────────────

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem efficiently and clearly. "
    "The last line of your response should be of the following format: "
    "'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' "
    "(without quotes) where ANSWER is just the final number or expression "
    "that solves the problem. Think step by step before answering.\n\n{problem}"
)

TEST_PROBLEMS = [
    {
        "name": "Number Theory — integer bases and divisibility",
        "problem": "Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.",
        "answer": 70,
    },
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
    {
        "name": "Geometry — triangle area from coordinates",
        "problem": (
            "On $\\triangle ABC$ points $A,D,E$, and $B$ lie that order on side $\\overline{AB}$ with $AD=4, DE=16$, and $EB=8$. "
            "Points $A,F,G$, and $C$ lie in that order on side $\\overline{AC}$ with $AF=13, FG=52$, and $GC=26$. "
            "Let $M$ be the reflection of $D$ through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has area 288. Find the area of heptagon $AFNBCEM$."
        ),
        "answer": 588,
    },
    {
        "name":"Algebra — integer solutions to quadratic form",
        "problem":"Find the number of ordered pairs $(x,y)$, where both $x$ and $y$ are integers between $-100$ and $100$, inclusive, such that $12x^{2}-xy-6y^{2}=0$.",
        "answer": 117,
    }
    # add problem 4 AIME
]


def format_prompt(problem: str, tokenizer, thinking: bool = False) -> str:
    query = MATH_PROMPT_TEMPLATE.format(problem=problem)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = []
            if thinking:
                messages.append({"role": "system", "content": "\nthinking on\n"})
            messages.append({"role": "user", "content": query})
            print(f"Input prompt with chat template applied: {tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}")
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return query


# ── One run ───────────────────────────────────────────────────────────

def run_one(
    llm,
    prompt: str,
    n_particles: int,
    alpha: float,
    ess_threshold: float,
    max_tokens: int,
    seed: int | None,
    label: str,
    alpha_ramp_tokens: int = 0,
) -> dict[str, Any]:
    """Run generation with one ESS threshold setting; return full metrics."""
    from vllm import SamplingParams

    # Seed Python's random so systematic_resample is deterministic per run.
    if seed is not None:
        random.seed(seed)

    sp = SamplingParams(
        n=n_particles,
        temperature=1.0 / alpha, # proposal temperature (not the same as SMC alpha) optimal proposal: q*(v) ∝ p(v)^α
        smc_alpha=alpha,
        smc_ess_threshold=ess_threshold,
        smc_alpha_ramp_tokens=alpha_ramp_tokens if alpha_ramp_tokens > 0 else None,
        max_tokens=max_tokens,
        logprobs=0,
        top_p=1.0,
        seed=seed,
    )

    print(f"\n  [{label}] n={n_particles}, α={alpha}, τ={ess_threshold}, "
          f"seed={seed}")

    instr = SMCInstrumentation()
    t_start = time.time()
    with instr:
        outputs = llm.generate([prompt], sp, use_tqdm=True)
    gen_time = time.time() - t_start

    print(f"  [{label}] Done in {gen_time:.1f}s | "
          f"{len(instr.steps)} steps | "
          f"{len(instr.resample_events)} resampling event(s)")

    request_output = outputs[0]
    completions = request_output.outputs

    # Map completions to particle indices (sorted by index in ID)
    # After output remapping, IDs are back to original "{idx}_{parent_id}"
    answers = []
    token_counts = []
    texts = []
    for comp in completions:
        texts.append(comp.text)
        token_counts.append(len(comp.token_ids))
        answers.append(extract_aime_answer(comp.text))

    # Final per-slot weights from SMCController.get_final_weights():
    # frozen weight for finished particles, current accumulated weight for
    # active ones. This is correct after resampling (no double-counting).
    weight_vec = [0.0] * n_particles
    if instr._controller is not None:
        for parent_id in instr._controller._groups:
            final_w = instr._controller.get_final_weights(parent_id)
            for slot_idx, w in final_w.items():
                if 0 <= slot_idx < n_particles:
                    weight_vec[slot_idx] = w

    # ESS metrics
    ess_traj = instr.ess_trajectory()
    min_ess = instr.min_ess()
    mean_ess = instr.mean_ess()
    final_ess = instr.final_ess()

    # Voting
    mv = majority_vote(answers)
    wv = weighted_majority_vote(answers, weight_vec)
    snis = snis_draw(answers, weight_vec, rng=torch.Generator().manual_seed(seed) if seed is not None else None)

    unique_answers = len({a for a in answers if a is not None})
    n_correct = sum(1 for a in answers if a is not None and a != 0)  # placeholder

    return {
        "label": label,
        "config": {
            "n_particles": n_particles,
            "alpha": alpha,
            "ess_threshold": ess_threshold,
            "max_tokens": max_tokens,
            "seed": seed,
        },
        "texts": texts,
        "answers": answers,
        "token_counts": token_counts,
        "weight_vec": weight_vec,
        "ess_trajectory": ess_traj,
        "min_ess": min_ess,
        "mean_ess": mean_ess,
        "final_ess": final_ess,
        "n_resampling_events": len(instr.resample_events),
        "resampling_events": instr.resample_events,
        "majority_vote": mv,
        "weighted_vote": wv,
        "snis_draw": snis,
        "unique_answers": unique_answers,
        "gen_time_seconds": gen_time,
        "n_steps": len(instr.steps),
        "completions": [c.text for c in completions],
    }


# ── Comparison print ──────────────────────────────────────────────────

def print_comparison(
    gt: int,
    baseline: dict[str, Any],
    resampled: dict[str, Any],
) -> None:
    n = baseline["config"]["n_particles"]
    print(f"\n  {'─' * 62}")
    print(f"  {'Metric':<35} {'Baseline':>12} {'Resampling':>12}")
    print(f"  {'─' * 62}")

    def row(label, base_val, resamp_val, fmt=".4f"):
        b = format(base_val, fmt) if isinstance(base_val, float) else str(base_val)
        r = format(resamp_val, fmt) if isinstance(resamp_val, float) else str(resamp_val)
        print(f"  {label:<35} {b:>12} {r:>12}")

    row("ESS threshold (τ)",
        baseline["config"]["ess_threshold"],
        resampled["config"]["ess_threshold"])
    row("Steps (tokens generated / N)", baseline["n_steps"], resampled["n_steps"], "d")
    row("Resampling events", baseline["n_resampling_events"],
        resampled["n_resampling_events"], "d")
    row("Min ESS (× N particles)",
        baseline["min_ess"] * n, resampled["min_ess"] * n)
    row("Mean ESS (× N particles)",
        baseline["mean_ess"] * n, resampled["mean_ess"] * n)
    row("Final ESS (× N particles)",
        baseline["final_ess"] * n, resampled["final_ess"] * n)
    row("Unique final answers",
        baseline["unique_answers"], resampled["unique_answers"], "d")

    def vote_str(v): return f"{v} {'✓' if v == gt else '✗'}" if v is not None else "None"
    row("Majority vote",
        vote_str(baseline["majority_vote"]),
        vote_str(resampled["majority_vote"]))
    row("Weighted vote (cum. weights)",
        vote_str(baseline["weighted_vote"]),
        vote_str(resampled["weighted_vote"]))
    row("SNIS draw (cum. weights)",
        vote_str(baseline["snis_draw"]),
        vote_str(resampled["snis_draw"]))
    row("Gen time (s)",
        baseline["gen_time_seconds"], resampled["gen_time_seconds"], ".1f")
    print(f"  {'─' * 62}")

    # Per-particle detail for resampling run
    print(f"\n  Per-particle detail ({resampled['label']}):")
    print(f"  {'Idx':>3}  {'Answer':>8}  {'Tokens':>6}  {'Cum weight':>12}  {'OK?':>4}")
    for i in range(n):
        ans = resampled["answers"][i]
        ans_str = str(ans) if ans is not None else "None"
        mark = "✓" if ans == gt else ("✗" if ans is not None else "")
        w = resampled["weight_vec"][i]
        toks = resampled["token_counts"][i]
        print(f"  {i:>3}  {ans_str:>8}  {toks:>6}  {w:>12.4f}  {mark:>4}")

    # Resampling event log
    events = resampled["resampling_events"]
    if events:
        print(f"\n  Resampling events ({len(events)}):")
        for ev in events:
            print(f"    step {ev['step']:>5}: parent={ev['parent_id']} | "
                  f"losers={[_short(x) for x in ev['loser_ids']]} | "
                  f"new slots={[p['slot'] for p in ev['new_particles']]}")
    else:
        print(f"\n  No resampling events fired.")

    # ESS trajectory summary (min every 10% of trajectory)
    traj = resampled["ess_trajectory"]
    if len(traj) > 0:
        step_size = max(1, len(traj) // 10)
        checkpoints = list(range(0, len(traj), step_size)) + [len(traj) - 1]
        checkpoints = sorted(set(checkpoints))
        print(f"\n  ESS trajectory [{resampled['label']}] "
              f"(N={n}, every ~{step_size} steps):")
        print("  " + "  ".join(f"s{traj[c]*n:.1f}" for c in checkpoints))


def _short(req_id: str) -> str:
    """Shorten a request ID for display."""
    parts = req_id.split("_")
    return parts[0] if parts else req_id


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Resampling Benchmark: live SMC resampling vs baseline"
    )
    parser.add_argument(
        "--model", type=str,
        default="/leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16",
    )
    parser.add_argument("--n_particles", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--alpha_ramp_tokens", type=int, default=0,
                        help="Ramp α from 1.0 to alpha over this many tokens (0=disabled)")
    parser.add_argument("--ess_threshold", type=float, default=0.5,
                        help="ESS threshold for live resampling (must be in (0,1))")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--problem_idx", type=int, default=2,
                        help="0=combinatorics, 1=number theory, 2=algebra (default)")
    parser.add_argument("--all_problems", action="store_true")
    parser.add_argument("--compare_baseline", action="store_true",
                        help="Also run a no-resampling baseline for comparison")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default="./smc_results")
    args = parser.parse_args()

    print("=" * 70)
    print("  Power-SMC Phase 2 Resampling Benchmark")
    print("  Live mid-generation SMC resampling via prefix-cache KV reuse")
    print("=" * 70)

    # Validate threshold
    if not (0.0 < args.ess_threshold < 1.0):
        parser.error(f"--ess_threshold must be in (0, 1), got {args.ess_threshold}")

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

    for problem in problems:
        print(f"\n{'═' * 70}")
        print(f"Problem: {problem['name']}")
        print(f"Ground truth: {problem['answer']}")
        print(f"{'═' * 70}")

        prompt = format_prompt(problem["problem"], tokenizer, thinking=args.thinking)
        prompt_len = len(tokenizer.encode(prompt))
        print(f"Prompt: {prompt_len} tokens | "
              f"n={args.n_particles}, α={args.alpha}, τ={args.ess_threshold}")

        problem_results: dict[str, Any] = {
            "problem": problem,
            "runs": {},
        }

        # ── Baseline run (no resampling) ──
        if args.compare_baseline:
            baseline_threshold = 0.001  # effectively never fires
            baseline = run_one(
                llm, prompt,
                n_particles=args.n_particles,
                alpha=args.alpha,
                ess_threshold=baseline_threshold,
                max_tokens=args.max_new_tokens,
                seed=args.seed,
                label="baseline",
                alpha_ramp_tokens=args.alpha_ramp_tokens,
            )
            problem_results["runs"]["baseline"] = baseline
        else:
            baseline = None

        # ── Resampling run ──
        resampled = run_one(
            llm, prompt,
            n_particles=args.n_particles,
            alpha=args.alpha,
            ess_threshold=args.ess_threshold,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
            label=f"τ={args.ess_threshold}",
            alpha_ramp_tokens=args.alpha_ramp_tokens,
        )
        problem_results["runs"]["resampled"] = resampled

        # ── Print results ──
        gt = problem["answer"]
        if baseline is not None:
            print_comparison(gt, baseline, resampled)
        else:
            # Print just the resampling run
            n = args.n_particles
            print(f"\n  ESS:  min={resampled['min_ess']*n:.2f}  "
                  f"mean={resampled['mean_ess']*n:.2f}  "
                  f"final={resampled['final_ess']*n:.2f}  "
                  f"(×{n} particles)")
            print(f"  Resampling events: {resampled['n_resampling_events']}")
            mv = resampled["majority_vote"]
            wv = resampled["weighted_vote"]
            snis = resampled["snis_draw"]
            print(f"  Majority vote:  {mv} {'✓' if mv == gt else '✗'}")
            print(f"  Weighted vote:  {wv} {'✓' if wv == gt else '✗'}")
            print(f"  SNIS draw:      {snis} {'✓' if snis == gt else '✗'}")
            print(f"  Ground truth:   {gt}")

            print(f"\n  Per-particle detail:")
            print(f"  {'Idx':>3}  {'Answer':>8}  {'Tokens':>6}  {'Cum weight':>12}")
            for i in range(n):
                ans = resampled["answers"][i]
                mark = "✓" if ans == gt else ("✗" if ans is not None else "")
                print(f"  {i:>3}  {str(ans) if ans is not None else 'None':>8}  "
                      f"{resampled['token_counts'][i]:>6}  "
                      f"{resampled['weight_vec'][i]:>12.4f}  {mark}")

            events = resampled["resampling_events"]
            if events:
                print(f"\n  Resampling events:")
                for ev in events:
                    print(f"    step {ev['step']:>5}: "
                          f"{len(ev['loser_ids'])} losers → "
                          f"{len(ev['new_particles'])} new particles | "
                          f"slots {[p['slot'] for p in ev['new_particles']]}")
            else:
                print(f"\n  No resampling events fired (ESS stayed above τ={args.ess_threshold}).")

            traj = resampled["ess_trajectory"]
            if traj:
                step_size = max(1, len(traj) // 10)
                checkpoints = sorted(set(
                    list(range(0, len(traj), step_size)) + [len(traj) - 1]
                ))
                print(f"\n  ESS×N trajectory (every ~{step_size} steps):")
                print("  " + "  ".join(f"s{traj[c]*n:.1f}" for c in checkpoints))

        print(f"\n  Time: {resampled['gen_time_seconds']:.1f}s")

        all_results.append(problem_results)

    # ── Save results ──
    out_name = (
        f"smc_benchmark_v4"
        f"_a{args.alpha}"
        f"_n{args.n_particles}"
        f"_tau{args.ess_threshold}"
        f"_s{args.seed}"
        f"_ramp{args.alpha_ramp_tokens}"
        f"{'_cmp' if args.compare_baseline else ''}"
        ".json"
    )
    output_file = Path(args.output_dir) / out_name

    def _serializable(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)
        return obj

    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items()
                    if k != "texts"}  # omit full generation text from JSON
        if isinstance(d, list):
            return [_clean(v) for v in d]
        if isinstance(d, float) and (math.isnan(d) or math.isinf(d)):
            return str(d)
        return d

    with open(output_file, "w") as f:
        json.dump(_clean(all_results), f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
