#!/usr/bin/env python3
"""Test script for Power-SMC Phases 1–3.

Run on a GPU compute node:
    python test_smc_phases.py --model /leonardo_scratch/fast/iGen_train/models/Domyn-Small-v0.2-bf16

Tests:
    Phase 1a — _compute_smc_weights math correctness (GPU)
    Phase 1b — SamplingParams validation + weight pipeline (no crash, weights flow)
    Phase 2a — SMCController unit: ESS, systematic resampling
    Phase 2b — SMCController integration: accumulate() called during inference
    Phase 3   — smc_log_weight field exists in CompletionOutput / API response
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from dataclasses import fields as dataclass_fields

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

results: list[tuple[str, str, str]] = []  # (name, status, detail)


def ok(name: str, detail: str = "") -> None:
    results.append((name, PASS, detail))
    print(f"  [{PASS}] {name}" + (f" — {detail}" if detail else ""))


def fail(name: str, detail: str) -> None:
    results.append((name, FAIL, detail))
    print(f"  [{FAIL}] {name} — {detail}")


def skip(name: str, detail: str) -> None:
    results.append((name, SKIP, detail))
    print(f"  [{SKIP}] {name} — {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1a — _compute_smc_weights GPU math
# ─────────────────────────────────────────────────────────────────────────────

def test_phase1a_smc_weights_math() -> None:
    print("\n=== Phase 1a: _compute_smc_weights correctness (GPU) ===")
    import torch
    from vllm.v1.sample.sampler import Sampler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        skip("weights_math_gpu", "no CUDA — running on CPU instead")

    # Known logits, alpha=2.0
    alpha = 2.0
    logits = torch.tensor(
        [[1.0, 2.0, 3.0, 0.5],
         [0.0, 0.0, 0.0, 0.0],   # uniform → logsumexp(2*log(0.25)*4) = log(4*0.25^2) = log(0.25)
         [10.0, -10.0, -10.0, -10.0]],  # near-greedy
        dtype=torch.float32,
        device=device,
    )

    # Reference: logsumexp(alpha * log_softmax(logits), dim=-1)
    expected = torch.logsumexp(alpha * logits.log_softmax(-1), dim=-1)

    smc_alphas = torch.tensor([alpha, alpha, alpha], dtype=torch.float32)  # CPU tensor

    class FakeMeta:
        def __init__(self):
            self.smc_alphas = smc_alphas

    result = Sampler._compute_smc_weights(logits, FakeMeta())

    if result is None:
        fail("weights_math_gpu", "_compute_smc_weights returned None unexpectedly")
        return

    if not torch.allclose(expected, result.cpu(), atol=1e-5):
        fail("weights_math_gpu", f"mismatch: expected={expected.tolist()}, got={result.cpu().tolist()}")
        return
    ok("weights_math_gpu", f"values={result.cpu().tolist()}")

    # Test None when smc_alphas is None
    class NoSMCMeta:
        def __init__(self):
            self.smc_alphas = None

    r2 = Sampler._compute_smc_weights(logits, NoSMCMeta())
    if r2 is not None:
        fail("weights_none_when_disabled", f"expected None, got {r2}")
    else:
        ok("weights_none_when_disabled")

    # Test non-SMC slots are zeroed
    smc_alphas_mixed = torch.tensor([alpha, 0.0, alpha], dtype=torch.float32)

    class MixedMeta:
        def __init__(self):
            self.smc_alphas = smc_alphas_mixed

    r3 = Sampler._compute_smc_weights(logits, MixedMeta())
    if r3 is None:
        fail("weights_zero_for_non_smc", "returned None for mixed alphas")
    elif r3[1].item() != 0.0:
        fail("weights_zero_for_non_smc", f"slot 1 (alpha=0) should be 0.0, got {r3[1].item()}")
    else:
        ok("weights_zero_for_non_smc", f"slot1={r3[1].item()}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1b — SamplingParams validation
# ─────────────────────────────────────────────────────────────────────────────

def test_phase1b_sampling_params() -> None:
    print("\n=== Phase 1b: SamplingParams SMC fields ===")
    from vllm.sampling_params import SamplingParams

    # Valid params
    try:
        sp = SamplingParams(n=4, smc_alpha=2.0, max_tokens=32)
        assert sp.smc_alpha == 2.0
        assert abs(sp.temperature - 0.5) < 1e-6, f"temperature should be 1/2=0.5, got {sp.temperature}"
        assert sp.smc_ess_threshold == 0.5
        assert sp.smc_alpha_ramp_tokens == 0
        ok("valid_params", f"alpha={sp.smc_alpha}, temp={sp.temperature}, ess={sp.smc_ess_threshold}")
    except Exception as e:
        fail("valid_params", str(e))

    # Custom ess threshold
    try:
        sp2 = SamplingParams(n=8, smc_alpha=3.0, smc_ess_threshold=0.3, max_tokens=32)
        assert sp2.smc_ess_threshold == 0.3
        ok("custom_ess_threshold", f"ess={sp2.smc_ess_threshold}")
    except Exception as e:
        fail("custom_ess_threshold", str(e))

    # n=1 should fail
    try:
        SamplingParams(n=1, smc_alpha=2.0, max_tokens=32)
        fail("n1_rejected", "should have raised ValueError")
    except ValueError as e:
        ok("n1_rejected", str(e))

    # alpha <= 1 should fail
    try:
        SamplingParams(n=4, smc_alpha=0.8, max_tokens=32)
        fail("alpha_le1_rejected", "should have raised ValueError")
    except ValueError as e:
        ok("alpha_le1_rejected", str(e))

    # alpha=1.0 should fail
    try:
        SamplingParams(n=4, smc_alpha=1.0, max_tokens=32)
        fail("alpha_eq1_rejected", "should have raised ValueError")
    except ValueError as e:
        ok("alpha_eq1_rejected", str(e))

    # bad ess_threshold
    try:
        SamplingParams(n=4, smc_alpha=2.0, smc_ess_threshold=1.5, max_tokens=32)
        fail("bad_ess_rejected", "should have raised ValueError")
    except ValueError as e:
        ok("bad_ess_rejected", str(e))

    # None = disabled (no n>1 requirement)
    try:
        sp3 = SamplingParams(n=1, smc_alpha=None, max_tokens=32)
        assert sp3.smc_alpha is None
        ok("disabled_when_none")
    except Exception as e:
        fail("disabled_when_none", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2a — SMCController unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_phase2a_controller_unit() -> None:
    print("\n=== Phase 2a: SMCController unit tests ===")
    from vllm.v1.engine.smc_controller import SMCController

    ctrl = SMCController()

    # ESS = 1.0 for uniform weights
    ess_uniform = ctrl.compute_ess([0.0, 0.0, 0.0, 0.0])
    if abs(ess_uniform - 1.0) > 1e-6:
        fail("ess_uniform", f"expected 1.0, got {ess_uniform}")
    else:
        ok("ess_uniform", f"ESS={ess_uniform:.4f}")

    # ESS → 1/N for degenerate (one particle dominates)
    n = 8
    lw_degen = [0.0] + [-100.0] * (n - 1)
    ess_degen = ctrl.compute_ess(lw_degen)
    expected_degen = 1.0 / n
    if abs(ess_degen - expected_degen) > 1e-4:
        fail("ess_degenerate", f"expected ~{expected_degen:.4f}, got {ess_degen:.4f}")
    else:
        ok("ess_degenerate", f"ESS={ess_degen:.4f} ≈ 1/N={expected_degen:.4f}")

    # Systematic resampling: uniform weights → ancestors should cover [0..N-1]
    ancestors = ctrl.systematic_resample([0.0, 0.0, 0.0, 0.0])
    if sorted(ancestors) != [0, 1, 2, 3]:
        fail("resample_uniform", f"expected [0,1,2,3], got {ancestors}")
    else:
        ok("resample_uniform", f"ancestors={ancestors}")

    # Resampling: degenerate weights → all ancestors = 0 (winner)
    ancestors_degen = ctrl.systematic_resample([0.0, -100.0, -100.0, -100.0])
    if not all(a == 0 for a in ancestors_degen):
        fail("resample_degenerate", f"expected all 0, got {ancestors_degen}")
    else:
        ok("resample_degenerate", f"ancestors={ancestors_degen}")

    # register_group / accumulate
    ctrl2 = SMCController()
    ctrl2.register_group("p1", ["c0", "c1", "c2", "c3"], alpha=2.0,
                         ess_threshold=0.5, alpha_ramp_tokens=0)
    assert "p1" in ctrl2._groups
    assert ctrl2._groups["p1"].log_weights == [0.0, 0.0, 0.0, 0.0]
    ok("register_group")

    ctrl2.accumulate({"c0": -1.5, "c1": -0.5, "c2": -1.5, "c3": -0.5})
    lw = ctrl2._groups["p1"].log_weights
    assert lw == [-1.5, -0.5, -1.5, -0.5], f"got {lw}"
    ok("accumulate_single_step", f"log_weights={lw}")

    ctrl2.accumulate({"c0": -1.0, "c1": -1.0, "c2": -1.0, "c3": -1.0})
    lw2 = ctrl2._groups["p1"].log_weights
    assert lw2 == [-2.5, -1.5, -2.5, -1.5], f"got {lw2}"
    ok("accumulate_two_steps", f"log_weights={lw2}")

    # step_count
    assert ctrl2._groups["p1"].step_count == 2
    ok("step_count", "step_count=2")

    # unregister_group
    ctrl2.unregister_group("p1")
    assert "p1" not in ctrl2._groups
    ok("unregister_group")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2b — Integration: weights flow through LLM.generate()
# ─────────────────────────────────────────────────────────────────────────────

def test_phase2b_integration(model_path: str) -> None:
    print("\n=== Phase 2b: Integration — weights flow during inference ===")

    import torch
    from unittest.mock import patch
    from vllm import LLM, SamplingParams
    from vllm.v1.engine.smc_controller import SMCController

    # Monkey-patch accumulate to capture what's passed in
    accumulated_weights: list[dict[str, float]] = []
    original_accumulate = SMCController.accumulate

    def capturing_accumulate(self, smc_log_weights):
        accumulated_weights.append(dict(smc_log_weights))
        return original_accumulate(self, smc_log_weights)

    print("  Loading model (this may take a minute)...")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.7,
    )
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    sp = SamplingParams(n=4, smc_alpha=2.0, max_tokens=20, temperature=0.5)

    with patch.object(SMCController, "accumulate", capturing_accumulate):
        prompt = "What is 2 + 2?"
        outputs = llm.generate([prompt], sp)

    # Check outputs structure
    if not outputs:
        fail("outputs_returned", "no outputs")
        return
    ok("outputs_returned", f"{len(outputs)} request output(s)")

    req_out = outputs[0]
    n_completions = len(req_out.outputs)
    if n_completions != 4:
        fail("n_completions", f"expected 4, got {n_completions}")
    else:
        ok("n_completions", f"n={n_completions}")

    for i, comp in enumerate(req_out.outputs):
        if not comp.token_ids:
            fail(f"completion_{i}_has_tokens", "empty token_ids")
        else:
            ok(f"completion_{i}_has_tokens", f"'{comp.text[:40]}...'")

    # Check weights were accumulated
    if not accumulated_weights:
        fail("weights_accumulated", "SMCController.accumulate never called — smc_log_weights dict was always empty/None")
    else:
        total_steps = len(accumulated_weights)
        sample_step = accumulated_weights[0]
        n_reqs_in_step = len(sample_step)
        all_vals = [v for d in accumulated_weights for v in d.values()]
        all_finite = all(math.isfinite(v) for v in all_vals)
        if not all_finite:
            fail("weights_finite", f"some weights are not finite: {all_vals[:8]}")
        else:
            ok("weights_accumulated",
               f"steps={total_steps}, reqs/step={n_reqs_in_step}, "
               f"sample={list(sample_step.values())[:4]}")

        # Weights should be negative (log of probability < 1)
        has_negative = any(v < 0 for v in all_vals)
        if not has_negative:
            fail("weights_negative", f"expected negative log-weights, got {all_vals[:4]}")
        else:
            ok("weights_negative", f"min={min(all_vals):.3f}, max={max(all_vals):.3f}")

    # Confirm no crash from SMCController hook in step()
    ok("no_crash_in_step")

    return llm  # return for reuse in phase 3 test


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — smc_log_weight field in outputs
# ─────────────────────────────────────────────────────────────────────────────

def test_phase3_output_fields(model_path: str, llm=None) -> None:
    print("\n=== Phase 3: smc_log_weight in output structures ===")

    # 3a: CompletionOutput has the field
    from vllm.outputs import CompletionOutput
    field_names = {f.name for f in dataclass_fields(CompletionOutput)}
    if "smc_log_weight" not in field_names:
        fail("completion_output_has_field", f"field missing; fields={field_names}")
    else:
        ok("completion_output_has_field")

    # 3b: EngineCoreOutput has the field
    from vllm.v1.engine import EngineCoreOutput
    import msgspec
    eco_fields = {f.name for f in msgspec.structs.fields(EngineCoreOutput)}
    if "smc_log_weight" not in eco_fields:
        fail("engine_core_output_has_field", f"field missing; fields={eco_fields}")
    else:
        ok("engine_core_output_has_field")

    # 3c: CompletionResponseChoice has the field
    from vllm.entrypoints.openai.completion.protocol import CompletionResponseChoice
    model_fields = set(CompletionResponseChoice.model_fields.keys())
    if "smc_log_weight" not in model_fields:
        fail("completion_response_choice_has_field", f"field missing; fields={model_fields}")
    else:
        ok("completion_response_choice_has_field")

    # 3d: End-to-end: smc_log_weight accessible on CompletionOutput objects
    # (currently None since EngineCoreOutput.smc_log_weight is not yet set by the scheduler)
    if llm is not None:
        from vllm import SamplingParams
        sp = SamplingParams(n=2, smc_alpha=2.0, max_tokens=10, temperature=0.5)
        outputs = llm.generate(["Hello"], sp)
        comp = outputs[0].outputs[0]
        # Field must exist and be accessible (None is OK at this stage)
        try:
            val = comp.smc_log_weight
            ok("smc_log_weight_accessible",
               f"value={val} (None expected until scheduler hook wires it)")
        except AttributeError as e:
            fail("smc_log_weight_accessible", str(e))
    else:
        skip("smc_log_weight_accessible", "no llm instance (model not loaded)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Power-SMC Phase 1-3 test suite")
    parser.add_argument(
        "--model",
        type=str,
        default="/leonardo_scratch/fast/iGen_train/models/Domyn-Small-v0.2-bf16",
        help="Model path (needed for integration tests)",
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Skip integration tests (no model load)",
    )
    args = parser.parse_args()

    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")

    # Phase 1a — unit, GPU math
    try:
        test_phase1a_smc_weights_math()
    except Exception as e:
        fail("phase1a_exception", traceback.format_exc())

    # Phase 1b — SamplingParams validation (no GPU needed)
    try:
        test_phase1b_sampling_params()
    except Exception as e:
        fail("phase1b_exception", traceback.format_exc())

    # Phase 2a — SMCController unit (no GPU needed)
    try:
        test_phase2a_controller_unit()
    except Exception as e:
        fail("phase2a_exception", traceback.format_exc())

    llm_instance = None
    if not args.unit_only:
        # Phase 2b — integration (requires GPU + model)
        try:
            llm_instance = test_phase2b_integration(args.model)
        except Exception as e:
            fail("phase2b_exception", traceback.format_exc())
            print(f"\n  (integration test failed — continuing with field-only Phase 3 tests)")
    else:
        skip("phase2b_integration", "--unit-only specified")

    # Phase 3 — output field checks + end-to-end
    try:
        test_phase3_output_fields(args.model, llm=llm_instance)
    except Exception as e:
        fail("phase3_exception", traceback.format_exc())

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for _, s, _ in results if "PASS" in s)
    n_fail = sum(1 for _, s, _ in results if "FAIL" in s)
    n_skip = sum(1 for _, s, _ in results if "SKIP" in s)
    for name, status, detail in results:
        suffix = f" — {detail}" if detail else ""
        print(f"  [{status}] {name}{suffix}")
    print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
