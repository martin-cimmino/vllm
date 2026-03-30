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
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

from smc_benchmark_v4 import (_clean_dict_for_json_formatting,
                              extract_aime_answer, format_prompt,
                              majority_vote, run_one)

# Force single-process mode so SMCInstrumentation monkey-patches work in-process
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


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
        problems.append(
            {
                "problem": row["problem"],
                "answer": int(row["answer"]),
            }
        )
    if max_problems is not None:
        problems = problems[:max_problems]
    return problems


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
    """Power-SMC run with live resampling; thin wrapper around v4's run_one()."""
    r = run_one(
        llm,
        prompt,
        n_particles=n_particles,
        alpha=alpha,
        ess_threshold=ess_threshold,
        max_tokens=max_tokens,
        seed=seed,
        label=f"τ={ess_threshold}",
        alpha_ramp_tokens=alpha_ramp_tokens,
    )
    r["method"] = "smc"
    r["n_correct"] = sum(1 for a in r["answers"] if a is not None and a == ground_truth)
    return r


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

    avg_at_k = mean(run["n_correct"] / k for gt, run in records)
    pass_at_k = mean(
        1.0 if any(a == gt for a in run["answers"] if a is not None) else 0.0
        for gt, run in records
    )
    total_gen_time = sum(run["gen_time_seconds"] for gt, run in records)
    mean_gen_time = total_gen_time / n

    majority_vote_acc = mean(
        1.0 if run.get("majority_vote") == gt else 0.0 for gt, run in records
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
            1.0 if run.get("majority_vote") == gt else 0.0 for gt, run in records
        )
        result["weighted_majority_acc"] = mean(
            1.0 if run.get("weighted_vote") == gt else 0.0 for gt, run in records
        )
        result["snis_draw_acc"] = mean(
            1.0 if run.get("snis_draw") == gt else 0.0 for gt, run in records
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
    parser.add_argument(
        "--n_particles",
        type=int,
        default=16,
        help="Number of particles (SMC) / samples (baseline)",
    )
    parser.add_argument(
        "--alpha", type=float, default=2.0, help="SMC twist parameter α"
    )
    parser.add_argument(
        "--ess_threshold",
        type=float,
        default=0.5,
        help="ESS threshold τ for live resampling (SMC only)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for baseline (default 1.0)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alpha_ramp_tokens",
        type=int,
        default=0,
        help="Ramp α from 1.0 over this many tokens (0=disabled)",
    )
    parser.add_argument(
        "--thinking", action="store_true", help="Enable thinking mode system message"
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Limit number of problems (for debugging)",
    )
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
        async_scheduling=False,  # TODO: IMPORTANT: ENFORCE this somewhere in the code when SMC is active
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
        print(
            f"  {problem['problem'][:120]}{'...' if len(problem['problem']) > 120 else ''}"
        )

        prompt = format_prompt(problem["problem"], tokenizer, thinking=args.thinking)
        prompt_len = len(tokenizer.encode(prompt))
        print(f"  Prompt: {prompt_len} tokens")

        problem_result: dict[str, Any] = {
            "problem_idx": idx,
            "problem": problem,
            "runs": {},
        }

        if run_baseline_flag:
            print(
                f"  [baseline] n={args.n_particles}, temp={args.temperature} ...",
                flush=True,
            )
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
    print(
        f"  Aggregate Metrics ({len(all_results)} problems, {total_elapsed:.1f}s total)"
    )
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
        print(
            f"    avg@{args.n_particles}  = {bm['avg_at_k']:.3f}  (expected acc of one random draw)"
        )
        print(
            f"    pass@{args.n_particles} = {bm['pass_at_k']:.3f}  (oracle: any sample correct)"
        )
        print(f"    majority_vote_acc = {bm['majority_vote_acc']:.3f}")

    if run_smc_flag:
        sm = compute_aggregate_metrics(all_results, "smc", args.n_particles)
        summary["smc"] = sm
        print(
            f"\n  SMC (n={args.n_particles}, α={args.alpha}, τ={args.ess_threshold}):"
        )
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

    output = {
        "summary": summary, 
        "details": _clean_dict_for_json_formatting(all_results),
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
