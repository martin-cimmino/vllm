# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 2b + Phase 3 end-to-end — Power-SMC integration tests.

Verifies that:
  - SMCController.accumulate() is called with finite, negative log-weights
    during a real inference run (Phase 2b).
  - n=4 SMC request produces exactly 4 completion outputs (Phase 2b).
  - smc_log_weight is accessible on CompletionOutput objects (Phase 3).

These tests load a small random model and require a CUDA device.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from contextlib import contextmanager
from collections.abc import Iterator
from unittest.mock import patch

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.v1.engine.smc_controller import SMCController

if not current_platform.is_cuda():
    pytest.skip(
        reason="V1 currently only supported on CUDA.",
        allow_module_level=True,
    )

# On the Leonardo cluster (no outbound internet on compute nodes) set
# SMC_TEST_MODEL to the local model path, e.g.:
#   export SMC_TEST_MODEL=/leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16
# Defaults to the small HF model used in open CI.
MODEL = os.environ.get(
    "SMC_TEST_MODEL",
    "hmellor/tiny-random-LlamaForCausalLM",
)
PROMPT = "What is 2 + 2?"

logger = logging.getLogger(__name__)


def _progress(msg: str) -> None:
    """Write directly to /dev/tty, bypassing pytest's fd-level capture.

    Falls back to stdout when there is no controlling terminal (e.g. SLURM
    batch jobs), where stdout is already a plain file and not captured.
    """
    try:
        with open("/dev/tty", "w") as tty:
            print(msg, file=tty, flush=True)
    except OSError:
        print(msg, file=sys.stdout, flush=True)


@contextmanager
def _heartbeat(label: str, interval: float = 15.0) -> Iterator[None]:
    """Print a still-running line every `interval` seconds while block runs."""
    stop = threading.Event()
    t0 = time.perf_counter()

    def _run() -> None:
        while not stop.wait(timeout=interval):
            _progress(f"  [smc_e2e] {label} still running... ({time.perf_counter() - t0:.0f}s)")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


@pytest.fixture(scope="module")
def llm() -> LLM:
    _progress(f"[smc_e2e] Loading model: {MODEL}")
    t0 = time.perf_counter()
    model = LLM(
        MODEL,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=1024,
        gpu_memory_utilization=0.9,
    )
    _progress(f"[smc_e2e] Model loaded in {time.perf_counter() - t0:.1f}s")
    return model


# ─── Phase 2b: weights flow during inference ─────────────────────────────────


@pytest.mark.skip_global_cleanup
def test_smc_accumulate_called_during_inference(llm: LLM) -> None:
    """SMCController.accumulate() is invoked with finite negative log-weights."""
    sp = SamplingParams(n=4, smc_alpha=2.0, max_tokens=5)
    accumulated: list[dict[str, float]] = []
    original = SMCController.accumulate

    def capturing(self, smc_log_weights: dict[str, float]) -> None:
        accumulated.append(dict(smc_log_weights))
        if len(accumulated) % 5 == 0:
            _progress(
                f"  [smc_e2e] accumulate() step {len(accumulated)}: "
                + str({k: f"{v:.4f}" for k, v in smc_log_weights.items()})
            )
        return original(self, smc_log_weights)

    _progress("[smc_e2e] Running accumulate test (n=4, max_tokens=5)...")
    t0 = time.perf_counter()
    with patch.object(SMCController, "accumulate", capturing), _heartbeat("accumulate test"):
        llm.generate([PROMPT], sp)
    elapsed = time.perf_counter() - t0
    _progress(f"[smc_e2e] generate() finished in {elapsed:.1f}s; accumulate() called {len(accumulated)} times")

    assert accumulated, "SMCController.accumulate() was never called"

    all_vals = [v for d in accumulated for v in d.values()]
    assert all(math.isfinite(v) for v in all_vals), (
        f"non-finite weights detected: {all_vals[:8]}"
    )
    assert any(v < 0 for v in all_vals), (
        f"expected negative log-weights (log-probs ≤ 0), got {all_vals[:8]}"
    )


@pytest.mark.skip_global_cleanup
def test_smc_n_completions(llm: LLM) -> None:
    """n=4 SMC request returns exactly 4 completion outputs."""
    _progress("[smc_e2e] Running n_completions test (n=4, max_tokens=10)...")
    sp = SamplingParams(n=4, smc_alpha=2.0, max_tokens=10)
    t0 = time.perf_counter()
    with _heartbeat("n_completions test"):
        outputs = llm.generate([PROMPT], sp)
    _progress(f"[smc_e2e] generate() finished in {time.perf_counter() - t0:.1f}s")
    assert len(outputs[0].outputs) == 4


@pytest.mark.skip_global_cleanup
def test_smc_completions_have_tokens(llm: LLM) -> None:
    """Each completion in a SMC request contains at least one token."""
    sp = SamplingParams(n=4, smc_alpha=2.0, max_tokens=10)
    outputs = llm.generate([PROMPT], sp)
    for i, comp in enumerate(outputs[0].outputs):
        logger.debug("[smc_e2e] completion %d: %d tokens", i, len(comp.token_ids))
        assert comp.token_ids, f"completion {i} has no tokens"


# ─── Phase 3 e2e: smc_log_weight on CompletionOutput ─────────────────────────


@pytest.mark.skip_global_cleanup
def test_smc_log_weight_attribute_accessible(llm: LLM) -> None:
    """smc_log_weight attribute exists on CompletionOutput (None is acceptable)."""
    sp = SamplingParams(n=2, smc_alpha=2.0, max_tokens=10)
    outputs = llm.generate(["Hello"], sp)
    comp = outputs[0].outputs[0]
    # AttributeError would fail this test; None value is expected until the
    # scheduler hook is wired to propagate the field.
    logger.debug("[smc_e2e] smc_log_weight=%r", comp.smc_log_weight)
    _ = comp.smc_log_weight
