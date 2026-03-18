#!/usr/bin/env python3
"""Phase 4 Multi-Problem Benchmark: Power-SMC vs Baseline on AIME 2025.

Runs baseline (standard sampling) and/or Power-SMC (live resampling) on all
30 AIME 2025 problems, computing aggregate metrics:
  - Baseline: avg@k (≈ pass@1), pass@k (oracle/best-of-k)
  - SMC: majority_vote_acc, weighted_majority_acc, snis_draw_acc

Usage:
    # Both methods, 16 particles, debug limit 3 problems:
    python smc_benchmark_v5.py --method both --max_problems 3 --n_particles 16 --alpha 2.0 --ess_threshold 0.5 --max_new_tokens 8192 --seed 42 --thinking --alpha_ramp_tokens 100
    python smc_benchmark_v5.py --method smc --max_problems 3 --n_particles 16 --alpha 3.0 --alpha_ramp_tokens 100 --ess_threshold 0.5 --max_new_tokens 16384 --seed 42 --thinking

    # Full AIME 2025 run (all 30 problems), SMC only:
    python smc_benchmark_v5.py --method smc --n_particles 16 \\
        --alpha 2.0 --ess_threshold 0.5 --max_new_tokens 8192

    # Via SLURM:
    sbatch run_smc_benchmark_v5.sbatch -- --method both --n_particles 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch

# Force single-process mode so SMCInstrumentation monkey-patches work in-process
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# export HF_DATASETS_CACHE=/leonardo_scratch/fast/AIFAC_L13_018/mcimmino/hf_cache
os.environ.setdefault("HF_DATASETS_CACHE", "/leonardo_scratch/fast/AIFAC_L13_018/mcimmino/hf_cache")
# export HF_DATASETS_OFFLINE=1
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# ── AIME 2025 Dataset Loading ─────────────────────────────────────────


def load_aime_2025(max_problems: int | None = None) -> list[dict[str, Any]]:
    """Load AIME 2025 problems from HuggingFace dataset.

    Uses HF_HOME cache — pre-download on a login node if needed:
        python -c "from datasets import load_dataset; load_dataset('yentinglin/aime_2025', split='train')"
    """
    from datasets import load_dataset

    ds = load_dataset("yentinglin/aime_2025", split="train")
    problems = []
    for row in ds:
        problems.append({
            "problem": row["problem"],
            "answer": int(row["answer"]),
        })
    if max_problems is not None:
        problems = problems[:max_problems]
    return problems


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
    raw = raw.replace(",", "").replace(" ", "").replace("\u2212", "-")
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


def weighted_majority_vote(
    answers: list[int | None], log_weights: list[float]) -> int | None:
    t = torch.tensor(log_weights, dtype=torch.float32)
    weights = torch.softmax(t, dim=0).tolist()
    vote: dict[int, float] = {}
    for ans, w in zip(answers, weights):
        if ans is not None:
            vote[ans] = vote.get(ans, 0.0) + w
    return max(vote, key=vote.__getitem__) if vote else None  # type: ignore[arg-type]


def snis_draw(
    answers: list[int | None],
    log_weights: list[float],
    rng: torch.Generator | None = None,) -> int | None:
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
        self.steps: list[dict[str, Any]] = []
        self.resample_events: list[dict[str, Any]] = []
        self._cum_weights: dict[str, float] = {}
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
            result = instr._orig_accumulate(ctrl_self, smc_log_weights)
            group_ess: dict[str, float] = {}
            for pid, group in ctrl_self._groups.items():
                ess = ctrl_self.compute_ess(group.log_weights)
                group_ess[pid] = ess
            instr.steps.append({
                "step": step_idx,
                "req_weights": dict(smc_log_weights),
                "group_ess": group_ess,
            })
            return result

        def patched_maybe_resample(ctrl_self, requests) -> dict:
            step_idx = len(instr.steps) - 1
            actions = instr._orig_maybe_resample(ctrl_self, requests)
            for pid, action in actions.items():
                event = {
                    "step": step_idx,
                    "parent_id": pid,
                    "loser_ids": list(action.loser_request_ids),
                    "n_new_particles": len(action.new_particles),
                }
                instr.resample_events.append(event)
                instr._id_to_original.update(ctrl_self._id_to_original)
            return actions

        SMCController.accumulate = patched_accumulate
        SMCController.maybe_resample = patched_maybe_resample
        return self

    def __exit__(self, *exc):
        from vllm.v1.engine.smc_controller import SMCController

        SMCController.accumulate = self._orig_accumulate
        SMCController.maybe_resample = self._orig_maybe_resample

    def ess_trajectory(self) -> list[float]:
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

    def cumulative_weights_by_original_id(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for req_id, w in self._cum_weights.items():
            original = self._id_to_original.get(req_id, req_id)
            totals[original] = totals.get(original, 0.0) + w
        return totals


# ── Prompt Template ───────────────────────────────────────────────────


MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem efficiently and clearly. "
    "The last line of your response should be of the following format: "
    "'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' "
    "(without quotes) where ANSWER is just the final number or expression "
    "that solves the problem. Think step by step before answering.\n\n{problem}"
)


def format_prompt(problem: str, tokenizer, thinking: bool = False) -> str:
    query = MATH_PROMPT_TEMPLATE.format(problem=problem)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = []
            if thinking:
                print("[DEBUG] Enabling thinking mode for prompt template.", flush=True)
                messages.append({"role": "system", "content": "\nthinking on\n"})
            messages.append({"role": "user", "content": query})
            print(f"Input prompt with chat template applied: {tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}")
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return query


# ── Baseline Run ──────────────────────────────────────────────────────


def run_baseline(
    llm,
    prompt: str,
    n: int,
    temperature: float,
    max_tokens: int,
    seed: int,
    ground_truth: int,
) -> dict[str, Any]:
    """Standard sampling: n independent draws, no SMC."""
    from vllm import SamplingParams

    sp = SamplingParams(
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        seed=seed,
    )

    t0 = time.time()
    outputs = llm.generate([prompt], sp, use_tqdm=False)
    gen_time = time.time() - t0

    completions = outputs[0].outputs
    answers = [extract_aime_answer(c.text) for c in completions]
    n_correct = sum(1 for a in answers if a is not None and a == ground_truth)

    # Uniform weights for reporting purposes (no SMC weighting)
    log_weights = [0.0] * n  # log(1/n) + const → uniform after softmax

    return {
        "method": "baseline",
        "config": {
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        },
        "answers": answers,
        "n_correct": n_correct,
        "majority_vote": majority_vote(answers),
        "gen_time_seconds": gen_time,
        "unique_answers": len({a for a in answers if a is not None}),
        "completions": [c.text for c in completions],
    }


# ── SMC Run ───────────────────────────────────────────────────────────


def run_smc(
    llm,
    prompt: str,
    n_particles: int,
    alpha: float,
    ess_threshold: float,
    max_tokens: int,
    seed: int,
    alpha_ramp_tokens: int,
    ground_truth: int,
) -> dict[str, Any]:
    """Power-SMC run with live resampling."""
    from vllm import SamplingParams

    # Seed Python's random so systematic_resample is deterministic per run.
    random.seed(seed)

    sp = SamplingParams(
        n=n_particles,
        temperature=1.0 / alpha,  # optimal proposal: q*(v) ∝ p(v)^α
        smc_alpha=alpha,
        smc_ess_threshold=ess_threshold,
        smc_alpha_ramp_tokens=alpha_ramp_tokens if alpha_ramp_tokens > 0 else None,
        max_tokens=max_tokens,
        logprobs=0,
        top_p=1.0,
        seed=seed,
    )

    instr = SMCInstrumentation()
    t0 = time.time()
    with instr:
        outputs = llm.generate([prompt], sp, use_tqdm=False)
    gen_time = time.time() - t0

    request_output = outputs[0]
    completions = request_output.outputs

    answers = [extract_aime_answer(c.text) for c in completions]
    n_correct = sum(1 for a in answers if a is not None and a == ground_truth)

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

    mv = majority_vote(answers)
    wv = weighted_majority_vote(answers, weight_vec)
    rng = torch.Generator().manual_seed(seed) if seed is not None else None
    sd = snis_draw(answers, weight_vec, rng=rng)

    min_ess_val = instr.min_ess()
    mean_ess_val = instr.mean_ess()

    return {
        "method": "smc",
        "config": {
            "n_particles": n_particles,
            "alpha": alpha,
            "ess_threshold": ess_threshold,
            "max_tokens": max_tokens,
            "seed": seed,
            "alpha_ramp_tokens": alpha_ramp_tokens,
        },
        "answers": answers,
        "weight_vec": weight_vec,
        "n_correct": n_correct,
        "majority_vote": mv,
        "weighted_vote": wv,
        "snis_draw": sd,
        "min_ess": min_ess_val if not math.isnan(min_ess_val) else None,
        "mean_ess": mean_ess_val if not math.isnan(mean_ess_val) else None,
        "n_resampling_events": len(instr.resample_events),
        "n_steps": len(instr.steps),
        "gen_time_seconds": gen_time,
        "unique_answers": len({a for a in answers if a is not None}),
        "completions": [c.text for c in completions],
    }


# ── Aggregate Metrics ─────────────────────────────────────────────────


def compute_aggregate_metrics(
    all_results: list[dict[str, Any]],
    method: str,
    n_particles: int,
) -> dict[str, Any]:
    records = [
        (r["problem"]["answer"], r["runs"][method])
        for r in all_results
        if method in r["runs"]
    ]
    n = len(records)
    if n == 0:
        return {"method": method, "n_problems": 0}

    n_key = "n" if method == "baseline" else "n_particles"
    k = records[0][1]["config"][n_key]

    avg_at_k = mean(
        run["n_correct"] / k for gt, run in records
    )
    pass_at_k = mean(
        1.0 if any(a == gt for a in run["answers"] if a is not None) else 0.0
        for gt, run in records
    )
    total_gen_time = sum(run["gen_time_seconds"] for gt, run in records)
    mean_gen_time = total_gen_time / n

    majority_vote_acc = mean(
        1.0 if run.get("majority_vote") == gt else 0.0
        for gt, run in records
    )

    result: dict[str, Any] = {
        "method": method,
        "n_problems": n,
        "k": k,
        "avg_at_k": avg_at_k,
        "pass_at_k": pass_at_k,
        "majority_vote_acc": majority_vote_acc,
        "total_gen_time_seconds": total_gen_time,
        "mean_gen_time_per_problem_seconds": mean_gen_time,
    }

    if method == "smc":
        result["majority_vote_acc"] = mean(
            1.0 if run.get("majority_vote") == gt else 0.0
            for gt, run in records
        )
        result["weighted_majority_acc"] = mean(
            1.0 if run.get("weighted_vote") == gt else 0.0
            for gt, run in records
        )
        result["snis_draw_acc"] = mean(
            1.0 if run.get("snis_draw") == gt else 0.0
            for gt, run in records
        )
        ess_vals = [
            run["min_ess"] * n_particles
            for gt, run in records
            if run.get("min_ess") is not None
        ]
        result["mean_min_ess_x_n"] = mean(ess_vals) if ess_vals else None
        result["mean_resampling_events"] = mean(
            run["n_resampling_events"] for gt, run in records
        )
        result["total_resampling_events"] = sum(
            run["n_resampling_events"] for gt, run in records
        )

    return result


# ── JSON Serialization ────────────────────────────────────────────────


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_clean(v) for v in d]
    if isinstance(d, float) and (math.isnan(d) or math.isinf(d)):
        return str(d)
    return d


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Multi-Problem SMC Benchmark on AIME 2025"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16",
    )
    parser.add_argument(
        "--method",
        choices=["baseline", "smc", "both"],
        default="both",
        help="Which method(s) to run",
    )
    parser.add_argument("--n_particles", type=int, default=16,
                        help="Number of particles (SMC) / samples (baseline)")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="SMC twist parameter α")
    parser.add_argument("--ess_threshold", type=float, default=0.5,
                        help="ESS threshold τ for live resampling (SMC only)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for baseline (default 1.0)")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha_ramp_tokens", type=int, default=0,
                        help="Ramp α from 1.0 over this many tokens (0=disabled)")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode system message")
    parser.add_argument("--max_problems", type=int, default=None,
                        help="Limit number of problems (for debugging)")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default="./smc_results")
    args = parser.parse_args()

    run_baseline_flag = args.method in ("baseline", "both")
    run_smc_flag = args.method in ("smc", "both")

    print("=" * 70)
    print("  Power-SMC Phase 4: Multi-Problem AIME 2025 Benchmark")
    print("=" * 70)
    print(f"  Method:        {args.method}")
    print(f"  n_particles:   {args.n_particles}")
    if run_smc_flag:
        print(f"  alpha:         {args.alpha}")
        print(f"  ess_threshold: {args.ess_threshold}")
    if run_baseline_flag:
        print(f"  temperature:   {args.temperature}")
    print(f"  max_new_tokens:{args.max_new_tokens}")
    print(f"  seed:          {args.seed}")
    print()

    # Load model
    print(f"Loading model: {args.model}")
    from vllm import LLM

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_new_tokens + 512,  # extra headroom for prompt
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    # Load problems
    print("Loading AIME 2025 problems from HuggingFace dataset...")
    problems = load_aime_2025(max_problems=args.max_problems)
    print(f"  {len(problems)} problems loaded\n")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results: list[dict[str, Any]] = []

    t_total_start = time.time()

    for idx, problem in enumerate(problems):
        print(f"{'─' * 70}")
        print(f"Problem {idx + 1}/{len(problems)}  |  answer={problem['answer']}")
        print(f"  {problem['problem'][:120]}{'...' if len(problem['problem']) > 120 else ''}")

        prompt = format_prompt(problem["problem"], tokenizer, thinking=args.thinking)
        prompt_len = len(tokenizer.encode(prompt))
        print(f"  Prompt: {prompt_len} tokens")

        problem_result: dict[str, Any] = {
            "problem_idx": idx,
            "problem": problem,
            "runs": {},
        }

        if run_baseline_flag:
            print(f"  [baseline] n={args.n_particles}, temp={args.temperature} ...", flush=True)
            t0 = time.time()
            b = run_baseline(
                llm=llm,
                prompt=prompt,
                n=args.n_particles,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                seed=args.seed,
                ground_truth=problem["answer"],
            )
            print(
                f"  [baseline] done in {time.time() - t0:.1f}s | "
                f"n_correct={b['n_correct']}/{args.n_particles} | "
                f"majority={b['majority_vote']} "
                f"{'✓' if b['majority_vote'] == problem['answer'] else '✗'}"
            )
            problem_result["runs"]["baseline"] = b

        if run_smc_flag:
            print(
                f"  [smc] n={args.n_particles}, α={args.alpha}, τ={args.ess_threshold} ...",
                flush=True,
            )
            t0 = time.time()
            s = run_smc(
                llm=llm,
                prompt=prompt,
                n_particles=args.n_particles,
                alpha=args.alpha,
                ess_threshold=args.ess_threshold,
                max_tokens=args.max_new_tokens,
                seed=args.seed,
                alpha_ramp_tokens=args.alpha_ramp_tokens,
                ground_truth=problem["answer"],
            )
            gt = problem["answer"]
            print(
                f"  [smc] done in {time.time() - t0:.1f}s | "
                f"resample_events={s['n_resampling_events']} | "
                f"n_correct={s['n_correct']}/{args.n_particles} | "
                f"majority={s['majority_vote']} {'✓' if s['majority_vote'] == gt else '✗'} | "
                f"weighted={s['weighted_vote']} {'✓' if s['weighted_vote'] == gt else '✗'} | "
                f"snis={s['snis_draw']} {'✓' if s['snis_draw'] == gt else '✗'}"
            )
            problem_result["runs"]["smc"] = s

        all_results.append(problem_result)

    total_elapsed = time.time() - t_total_start

    # ── Aggregate metrics ──
    print(f"\n{'═' * 70}")
    print(f"  Aggregate Metrics ({len(all_results)} problems, {total_elapsed:.1f}s total)")
    print(f"{'═' * 70}")

    summary: dict[str, Any] = {
        "config": {
            "method": args.method,
            "n_particles": args.n_particles,
            "alpha": args.alpha if run_smc_flag else None,
            "ess_threshold": args.ess_threshold if run_smc_flag else None,
            "temperature": args.temperature if run_baseline_flag else None,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "alpha_ramp_tokens": args.alpha_ramp_tokens,
            "model": args.model,
            "n_problems": len(all_results),
        },
        "total_elapsed_seconds": total_elapsed,
    }

    if run_baseline_flag:
        bm = compute_aggregate_metrics(all_results, "baseline", args.n_particles)
        summary["baseline"] = bm
        print(f"\n  Baseline (n={args.n_particles}, temp={args.temperature}):")
        print(f"    avg@{args.n_particles}  = {bm['avg_at_k']:.3f}  (expected acc of one random draw)")
        print(f"    pass@{args.n_particles} = {bm['pass_at_k']:.3f}  (oracle: any sample correct)")
        print(f"    majority_vote_acc = {bm['majority_vote_acc']:.3f}")

    if run_smc_flag:
        sm = compute_aggregate_metrics(all_results, "smc", args.n_particles)
        summary["smc"] = sm
        print(f"\n  SMC (n={args.n_particles}, α={args.alpha}, τ={args.ess_threshold}):")
        print(f"    avg@{args.n_particles}               = {sm['avg_at_k']:.3f}")
        print(f"    pass@{args.n_particles}              = {sm['pass_at_k']:.3f}")
        print(f"    majority_vote_acc     = {sm['majority_vote_acc']:.3f}")
        print(f"    weighted_majority_acc = {sm['weighted_majority_acc']:.3f}")
        print(f"    snis_draw_acc         = {sm['snis_draw_acc']:.3f}")
        if sm.get("mean_min_ess_x_n") is not None:
            print(f"    mean_min_ESS×N        = {sm['mean_min_ess_x_n']:.2f}")
        print(f"    total_resampling_evts = {sm['total_resampling_events']}")
        print(f"    mean_resampling_evts  = {sm['mean_resampling_events']:.1f}")

    # ── Save results ──
    method_tag = args.method
    alpha_tag = f"_a{args.alpha}" if run_smc_flag else ""
    tau_tag = f"_tau{args.ess_threshold}" if run_smc_flag else ""
    out_name = (
        f"smc_benchmark_v5"
        f"_{method_tag}"
        f"{alpha_tag}"
        f"_n{args.n_particles}"
        f"{tau_tag}"
        f"_s{args.seed}"
        ".json"
    )
    output_file = Path(args.output_dir) / out_name

    output = {"summary": summary, "details": _clean(all_results)}
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
