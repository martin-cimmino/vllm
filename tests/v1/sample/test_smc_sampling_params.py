# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 1b — Power-SMC SamplingParams validation.

Tests that SamplingParams correctly validates and stores SMC-specific fields
(smc_alpha, smc_ess_threshold, smc_alpha_ramp_tokens) and auto-derives
temperature = 1/alpha.
"""
from __future__ import annotations

import pytest

from vllm.sampling_params import SamplingParams


def test_smc_params_stored_correctly() -> None:
    """Valid SMC params are accepted with correct defaults."""
    sp = SamplingParams(n=4, smc_alpha=2.0, max_tokens=32)
    assert sp.smc_alpha == 2.0
    assert abs(sp.temperature - 0.5) < 1e-6, (
        f"temperature should be 1/alpha=0.5, got {sp.temperature}"
    )
    assert sp.smc_ess_threshold == 0.5
    assert sp.smc_alpha_ramp_tokens == 0


def test_smc_custom_ess_threshold() -> None:
    """Custom ESS threshold is stored correctly."""
    sp = SamplingParams(n=8, smc_alpha=3.0, smc_ess_threshold=0.3, max_tokens=32)
    assert sp.smc_ess_threshold == 0.3


def test_smc_temperature_derived_from_alpha() -> None:
    """temperature is auto-set to 1/alpha for several alpha values."""
    for alpha in [2.0, 3.0, 5.0]:
        sp = SamplingParams(n=4, smc_alpha=alpha, max_tokens=32)
        assert abs(sp.temperature - 1.0 / alpha) < 1e-6, (
            f"alpha={alpha}: expected temperature={1/alpha}, got {sp.temperature}"
        )


@pytest.mark.parametrize("alpha", [0.5, 0.8, 1.0])
def test_smc_alpha_le1_rejected(alpha: float) -> None:
    """alpha <= 1 is invalid (alpha must be > 1 to sharpen the distribution)."""
    with pytest.raises(ValueError):
        SamplingParams(n=4, smc_alpha=alpha, max_tokens=32)


@pytest.mark.parametrize("ess", [-0.1, 0.0, 1.0, 1.5])
def test_smc_bad_ess_threshold_rejected(ess: float) -> None:
    """ESS threshold outside (0, 1) exclusive is rejected."""
    with pytest.raises(ValueError):
        SamplingParams(n=4, smc_alpha=2.0, smc_ess_threshold=ess, max_tokens=32)


def test_smc_disabled_when_alpha_none() -> None:
    """smc_alpha=None disables SMC; n=1 is permitted."""
    sp = SamplingParams(n=1, smc_alpha=None, max_tokens=32)
    assert sp.smc_alpha is None
