# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 1a — Power-SMC incremental weight computation in the Sampler.

Tests for Sampler._compute_smc_weights: verifies the
logsumexp(α·log_softmax(logits)) formula, None-passthrough when SMC is
disabled, and zeroing of non-SMC slots.
"""
from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.sample.sampler import Sampler

if not current_platform.is_cuda():
    pytest.skip(
        reason="SMC weight tests require CUDA.",
        allow_module_level=True,
    )

_DEVICE = f"{current_platform.device_type}:0"
_ALPHA = 2.0


class _FakeSamplingMeta:
    """Minimal stand-in for SamplingMetadata carrying SMC fields."""

    def __init__(self, smc_alphas: torch.Tensor | None) -> None:
        self.smc_alphas = smc_alphas
        self.smc_alpha_ramp_tokens = None
        self.smc_step_counts = None


def _make_logits() -> torch.Tensor:
    """Three-row logit matrix covering non-uniform, uniform, and near-greedy."""
    return torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.5],
            [0.0, 0.0, 0.0, 0.0],  # uniform → predictable logsumexp
            [10.0, -10.0, -10.0, -10.0],  # near-greedy
        ],
        dtype=torch.float32,
        device=_DEVICE,
    )


def test_smc_weights_values() -> None:
    """Computed weights match logsumexp(α·log_softmax(logits)) reference."""
    logits = _make_logits()
    expected = torch.logsumexp(_ALPHA * logits.log_softmax(-1), dim=-1)
    smc_alphas = torch.full((3,), _ALPHA, dtype=torch.float32, device=_DEVICE)

    result = Sampler._compute_smc_weights(logits, _FakeSamplingMeta(smc_alphas))

    assert result is not None
    assert torch.allclose(expected.cpu(), result.cpu(), atol=1e-5), (
        f"mismatch: expected={expected.cpu().tolist()}, got={result.cpu().tolist()}"
    )


def test_smc_weights_returns_none_when_disabled() -> None:
    """Returns None when smc_alphas is None (SMC disabled for all requests)."""
    result = Sampler._compute_smc_weights(
        _make_logits(), _FakeSamplingMeta(smc_alphas=None)
    )
    assert result is None


def test_smc_weights_zero_for_non_smc_slots() -> None:
    """Slots with alpha=0 (non-SMC requests) produce weight 0.0."""
    smc_alphas = torch.tensor([_ALPHA, 0.0, _ALPHA], dtype=torch.float32, device=_DEVICE)
    result = Sampler._compute_smc_weights(
        _make_logits(), _FakeSamplingMeta(smc_alphas)
    )
    assert result is not None
    assert result[1].item() == 0.0, (
        f"slot 1 (alpha=0) should be 0.0, got {result[1].item()}"
    )


@pytest.mark.parametrize("alpha", [2.0, 3.0, 5.0])
def test_smc_weights_different_alphas(alpha: float) -> None:
    """Weight formula is consistent across different alpha values."""
    logits = _make_logits()
    expected = torch.logsumexp(alpha * logits.log_softmax(-1), dim=-1)
    smc_alphas = torch.full((3,), alpha, dtype=torch.float32, device=_DEVICE)

    result = Sampler._compute_smc_weights(logits, _FakeSamplingMeta(smc_alphas))

    assert result is not None
    assert torch.allclose(expected.cpu(), result.cpu(), atol=1e-5)


def test_smc_weights_are_non_positive() -> None:
    """Incremental log-weights are ≤ 0 (they are log of a probability ≤ 1)."""
    smc_alphas = torch.full((3,), _ALPHA, dtype=torch.float32, device=_DEVICE)
    result = Sampler._compute_smc_weights(
        _make_logits(), _FakeSamplingMeta(smc_alphas)
    )
    assert result is not None
    assert (result <= 0).all(), f"some weights > 0: {result.tolist()}"
