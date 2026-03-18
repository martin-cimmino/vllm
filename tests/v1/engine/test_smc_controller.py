# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 2a — SMCController unit tests; Phase 3 — output field checks.

Phase 2a: verifies ESS computation, systematic resampling, the
register/accumulate/unregister lifecycle of SMCController, and the
Phase 2 ResampleAction construction (structured resampling output).

Phase 3 (static): verifies that smc_log_weight and smc_detokenizer_reset
are present in CompletionOutput, EngineCoreOutput, and
CompletionResponseChoice — no model loading required.
"""
from __future__ import annotations

import random
from dataclasses import fields as dataclass_fields
from types import SimpleNamespace

import msgspec
import pytest

from vllm.outputs import CompletionOutput
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine.smc_controller import (
    NewParticle,
    ResampleAction,
    SMCController,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_fake_request(req_id: str, all_tokens: list[int],
                       output_tokens: list[int] | None = None,
                       max_tokens: int | None = None):
    """Create a SimpleNamespace that quacks like a Request for testing."""
    if output_tokens is None:
        output_tokens = []
    # sampling_params=None triggers the group.original_max_tokens fallback
    # in maybe_resample; pass a namespace with max_tokens when you need a
    # specific budget to be respected across chained resamplings.
    sp = (
        SimpleNamespace(max_tokens=max_tokens)
        if max_tokens is not None
        else None
    )
    return SimpleNamespace(
        request_id=req_id,
        all_token_ids=list(all_tokens),
        _output_token_ids=list(output_tokens),
        sampling_params=sp,
    )


def _build_requests_dict(ctrl: SMCController, group_key: str,
                         prompt_tokens: list[int] | None = None,
                         output_len: int = 5) -> dict[str, object]:
    """Build a fake requests dict for the controller's group."""
    group = ctrl._groups[group_key]
    if prompt_tokens is None:
        prompt_tokens = [1, 2, 3]
    reqs: dict[str, object] = {}
    for rid in group.child_request_ids:
        out_toks = list(range(100, 100 + output_len))
        reqs[rid] = _make_fake_request(
            rid,
            all_tokens=prompt_tokens + out_toks,
            output_tokens=out_toks,
        )
    return reqs


# ─── Phase 2a: SMCController unit tests ──────────────────────────────────────


def test_ess_uniform() -> None:
    """Uniform log-weights → ESS = 1.0 (maximum diversity)."""
    ctrl = SMCController()
    ess = ctrl.compute_ess([0.0, 0.0, 0.0, 0.0])
    assert abs(ess - 1.0) < 1e-6, f"expected ESS=1.0, got {ess}"


@pytest.mark.parametrize("n", [4, 8, 16])
def test_ess_degenerate(n: int) -> None:
    """One dominant particle → ESS ≈ 1/N (minimum diversity)."""
    ctrl = SMCController()
    lw = [0.0] + [-100.0] * (n - 1)
    ess = ctrl.compute_ess(lw)
    expected = 1.0 / n
    assert abs(ess - expected) < 1e-4, (
        f"n={n}: expected ESS≈{expected:.4f}, got {ess:.4f}"
    )

def test_systematic_resample_uniform() -> None:
    """Uniform weights → ancestors are a permutation of [0, N-1]."""
    ctrl = SMCController()
    ancestors = ctrl.systematic_resample([0.0, 0.0, 0.0, 0.0])
    assert sorted(ancestors) == [0, 1, 2, 3]


def test_systematic_resample_balanced() -> None:
    """Balanced weights → ancestors reflect relative weights."""
    ctrl = SMCController()
    ancestors = ctrl.systematic_resample([-1.0, 0.0, -0.5, -2.0])
    assert sorted(ancestors) == [0, 1, 1, 2]


def test_systematic_resample_degenerate() -> None:
    """Degenerate weights → all ancestors equal the winning particle index."""
    ctrl = SMCController()
    ancestors = ctrl.systematic_resample([0.0, -100.0, -100.0, -100.0])
    assert all(a == 0 for a in ancestors), (
        f"expected all 0 (winner), got {ancestors}"
    )


def test_register_group_initialises_uniform_weights() -> None:
    """register_group creates a particle group with zero log-weights."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0,
    )
    assert "p1" in ctrl._groups
    assert ctrl._groups["p1"].log_weights == [0.0, 0.0, 0.0, 0.0]


def test_accumulate_single_step() -> None:
    """accumulate() sets log-weights on the first step."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0,
    )
    ctrl.accumulate({"c0": -1.5, "c1": -0.5, "c2": -1.5, "c3": -0.5})
    assert ctrl._groups["p1"].log_weights == [-1.5, -0.5, -1.5, -0.5]


def test_accumulate_is_additive() -> None:
    """accumulate() adds incremental weights across multiple steps."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0,
    )
    ctrl.accumulate({"c0": -1.5, "c1": -0.5, "c2": -1.5, "c3": -0.5})
    ctrl.accumulate({"c0": -1.0, "c1": -1.0, "c2": -1.0, "c3": -1.0})
    assert ctrl._groups["p1"].log_weights == [-2.5, -1.5, -2.5, -1.5]


def test_step_count_increments() -> None:
    """step_count is incremented once per accumulate() call."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1"], alpha=2.0, ess_threshold=0.5, alpha_ramp_tokens=0,
    )
    ctrl.accumulate({"c0": -1.0, "c1": -1.0})
    ctrl.accumulate({"c0": -1.0, "c1": -1.0})
    assert ctrl._groups["p1"].step_count == 2


def test_unregister_group_removes_entry() -> None:
    """unregister_group removes the particle group from the controller."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1"], alpha=2.0, ess_threshold=0.5, alpha_ramp_tokens=0,
    )
    ctrl.unregister_group("p1")
    assert "p1" not in ctrl._groups


# ─── Phase 2: ResampleAction construction ────────────────────────────────────


# active list : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# anc_pos:      [1, 1, 1, 1, 1, 4, 5, 6, 7, 8,  9, 10, 11, 12, 13, 14]

def test_maybe_resample_losers_identified_correctly() -> None:
    """ResampleAction.loser_request_ids correctly identifies losers based on ESS.

    With seed=42 and these 16-particle weights, ESS ≈ 0.499 < 0.5 → fires.
    anc_pos = [0, 0, 2, 3, 3, 3, 3, 4, 6, 7, 8, 9, 10, 12, 13, 15]
    True winners (self-mapped): slots 0, 2, 3, 15.
    Proxy ancestors (direct, not resolved): slots 4, 6, 7, 8, 9, 10, 12, 13.
    The abort-after-create fix in core.py ensures proxy ancestors are still
    alive when their token sequences are read.
    """
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9",
               "c10", "c11", "c12", "c13", "c14", "c15"],
        alpha=4.0, ess_threshold=0.5,
        alpha_ramp_tokens=100, original_max_tokens=8192,
    )
    ctrl.accumulate({
        "c0": -10.464439448678107, "c1": -12.235446452771413,
        "c2": -11.562239008591305, "c3": -9.39429838034851,
        "c4": -12.295605566207932, "c5": -11.807898694641466,
        "c6": -11.060544531512818, "c7": -11.101674714510736,
        "c8": -10.540749268157178, "c9": -11.52127701671975,
        "c10": -10.976022033243552, "c11": -11.155199924152399,
        "c12": -11.333260488761333, "c13": -11.167389545859358,
        "c14": -12.292045226215656, "c15": -10.63779145808848,
    })
    reqs = _build_requests_dict(ctrl, "p1")
    random.seed(42)
    actions = ctrl.maybe_resample(reqs)

    # With seed=42: anc_pos = [0,0,2,3, 3,3,3,4, 6,7,8,9, 10,12,13,15]
    # Winners (self-mapped slots): 0, 2, 3, 15.
    # Losers: all other 12 slots.
    losers = {"c1", "c4", "c5", "c6", "c7", "c8", "c9", "c10",
              "c11", "c12", "c13", "c14"}
    # Direct ancestors for each loser (proxy ancestors allowed with abort-after-create):
    # c1→c0, c4→c3, c5→c3, c6→c3, c7→c4, c8→c6, c9→c7,
    # c10→c8, c11→c9, c12→c10, c13→c12, c14→c13
    expected_ancestors = {
        "c1": "c0",  "c4": "c3",  "c5": "c3",  "c6": "c3",
        "c7": "c4",  "c8": "c6",  "c9": "c7",  "c10": "c8",
        "c11": "c9", "c12": "c10", "c13": "c12", "c14": "c13",
    }
    # Build a slot→original_id reverse map from child_request_ids (pre-resample)
    group = ctrl._groups["p1"]
    # The child_request_ids are updated in-place by maybe_resample, but
    # new_particles carry ancestor_request_id = group.child_request_ids[true_anc_slot]
    # which still equals the original cN names for all 16 slots here.

    assert "p1" in actions
    action = actions["p1"]
    assert set(action.loser_request_ids) == losers
    assert len(action.new_particles) == 12
    for particle in action.new_particles:
        assert isinstance(particle, NewParticle)
        # Recover the original loser ID via _id_to_original (which maps new_id→original)
        orig_loser = ctrl.get_original_id(particle.new_request_id)
        expected_anc = expected_ancestors[orig_loser]
        assert particle.ancestor_request_id == expected_anc, (
            f"loser {orig_loser}: expected ancestor {expected_anc}, "
            f"got {particle.ancestor_request_id}"
        )


def test_maybe_resample_no_action_when_ess_high() -> None:
    """No resample action when ESS is above threshold."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
    )
    # Uniform weights → ESS = 1.0 > 0.5
    ctrl.accumulate({"c0": -1.0, "c1": -1.0, "c2": -1.0, "c3": -1.0})
    reqs = _build_requests_dict(ctrl, "p1")
    actions = ctrl.maybe_resample(reqs)
    assert actions == {}


def test_maybe_resample_returns_action_when_degenerate() -> None:
    """Degenerate weights trigger resampling with correct structure."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
    )
    # Degenerate: c0 dominates → ESS ≈ 0.25 < 0.5
    ctrl.accumulate({"c0": 0.0, "c1": -100.0, "c2": -100.0, "c3": -100.0})
    reqs = _build_requests_dict(ctrl, "p1")
    actions = ctrl.maybe_resample(reqs)
    assert "p1" in actions
    action = actions["p1"]
    assert isinstance(action, ResampleAction)
    # c0 is the winner; c1, c2, c3 are losers
    assert set(action.loser_request_ids) == {"c1", "c2", "c3"}
    # 3 new particles (for slots 1, 2, 3 — slot 0 stays as c0)
    assert len(action.new_particles) == 3
    for p in action.new_particles:
        assert isinstance(p, NewParticle)
        assert p.ancestor_request_id == "c0"
        assert p.slot_index in (1, 2, 3)


def test_winner_keeps_own_slot() -> None:
    """Winner particle stays in its original slot (no new request created)."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
    )
    ctrl.accumulate({"c0": 0.0, "c1": -100.0, "c2": -100.0, "c3": -100.0})
    reqs = _build_requests_dict(ctrl, "p1")
    actions = ctrl.maybe_resample(reqs)
    action = actions["p1"]
    # Slot 0 should NOT appear in new_particles
    slot_indices = [p.slot_index for p in action.new_particles]
    assert 0 not in slot_indices


def test_weights_reset_after_resample() -> None:
    """Log-weights are reset to zero after resampling."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
    )
    ctrl.accumulate({"c0": 0.0, "c1": -100.0, "c2": -100.0, "c3": -100.0})
    reqs = _build_requests_dict(ctrl, "p1")
    ctrl.maybe_resample(reqs)
    assert ctrl._groups["p1"].log_weights == [0.0, 0.0, 0.0, 0.0]


def test_child_ids_updated_after_resample() -> None:
    """child_request_ids are updated to reflect new IDs for loser slots."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
    )
    ctrl.accumulate({"c0": 0.0, "c1": -100.0, "c2": -100.0, "c3": -100.0})
    reqs = _build_requests_dict(ctrl, "p1")
    ctrl.maybe_resample(reqs)
    group = ctrl._groups["p1"]
    # Slot 0 keeps original ID
    assert group.child_request_ids[0] == "c0"
    # Slots 1-3 get new IDs
    for i in (1, 2, 3):
        assert group.child_request_ids[i].startswith("smc_p1_")


def test_id_mapping_chains() -> None:
    """_id_to_original maps through multiple resampling events."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
    )
    # First resample
    ctrl.accumulate({"c0": 0.0, "c1": -100.0, "c2": -100.0, "c3": -100.0})
    reqs = _build_requests_dict(ctrl, "p1")
    ctrl.maybe_resample(reqs)

    # Verify first mapping: new IDs map to original child IDs
    for new_id in ctrl._groups["p1"].child_request_ids[1:]:
        assert ctrl.get_original_id(new_id) in {"c1", "c2", "c3"}

    # Second resample: rebuild requests with new IDs
    ctrl.accumulate({
        ctrl._groups["p1"].child_request_ids[0]: 0.0,
        ctrl._groups["p1"].child_request_ids[1]: -100.0,
        ctrl._groups["p1"].child_request_ids[2]: -100.0,
        ctrl._groups["p1"].child_request_ids[3]: -100.0,
    })
    reqs2 = _build_requests_dict(ctrl, "p1")
    ctrl.maybe_resample(reqs2)

    # All new IDs should still map to original c0-c3
    for new_id in ctrl._groups["p1"].child_request_ids:
        orig = ctrl.get_original_id(new_id)
        assert orig in {"c0", "c1", "c2", "c3"}, (
            f"{new_id} mapped to {orig}, not an original child ID"
        )


def test_max_tokens_in_new_particle() -> None:
    """NewParticle carries correct original_max_tokens and num_output_tokens."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=50,
    )
    ctrl.accumulate({"c0": 0.0, "c1": -100.0, "c2": -100.0, "c3": -100.0})
    reqs = _build_requests_dict(ctrl, "p1", output_len=10)
    actions = ctrl.maybe_resample(reqs)
    for p in actions["p1"].new_particles:
        assert p.original_max_tokens == 50
        assert p.num_output_tokens == 10
        # remaining = 50 - 10 = 40


def test_freezes_finished_particle_and_resamples_active() -> None:
    """When a particle finishes, ESS is computed over ALL N weights (zombie-
    inclusive). ESS collapse is driven by zombie vs active weight divergence.
    Resampling only touches active slots — zombie slots stay as-is.
    After resample, only active weights reset to 0; zombie log_weights are
    preserved at their frozen value."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
    )
    ctrl.accumulate({"c0": 0.0, "c1": -100.0, "c2": -100.0, "c3": -100.0})
    # c3 finished (missing from requests).
    reqs: dict[str, object] = {}
    for rid in ["c0", "c1", "c2"]:
        reqs[rid] = _make_fake_request(rid, [1, 2, 3, 100, 101], [100, 101])
    actions = ctrl.maybe_resample(reqs)

    group = ctrl._groups["p1"]

    # Resampling fires because ESS over all 4 weights (0, -100, -100, -100)
    # is ~0.25 < 0.5.
    assert "p1" in actions
    action = actions["p1"]

    # Active losers: c1 and c2 (slot 3 was zombie, no live request to abort).
    assert set(action.loser_request_ids) == {"c1", "c2"}

    # Zombie slot 3 is NOT included in the active-only resampling pool.
    # c0 dominates among active → slots 1 and 2 get NewParticles from c0.
    slot_indices = {p.slot_index for p in action.new_particles}
    assert slot_indices == {1, 2}
    assert 3 not in slot_indices
    assert action.zombie_clones == []

    # After resample: active weights (slots 0, 1, 2) reset to 0.
    # Zombie slot 3 frozen_weight is preserved (-100.0 from before finishing).
    assert group.log_weights[:3] == [0.0, 0.0, 0.0]
    assert group.log_weights[3] == pytest.approx(-100.0)
    assert 3 in group.frozen_weights


def test_all_particles_finished_skips_group() -> None:
    """Group is skipped (no resampling) when all particles have finished."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
    )
    ctrl.accumulate({"c0": 0.0, "c1": -100.0, "c2": -100.0, "c3": -100.0})
    # All particles finished — empty requests dict.
    actions = ctrl.maybe_resample({})
    assert actions == {}
    # All slots should be frozen.
    group = ctrl._groups["p1"]
    assert set(group.frozen_weights.keys()) == {0, 1, 2, 3}


def test_get_final_weights_merges_frozen_and_active() -> None:
    """get_final_weights() returns frozen weight for finished particles and
    current log_weight for still-active particles."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
    )
    ctrl.accumulate({"c0": -1.0, "c1": -2.0, "c2": -3.0, "c3": -4.0})
    # c3 is finished — freeze it by passing a requests dict without c3.
    reqs: dict[str, object] = {}
    for rid in ["c0", "c1", "c2"]:
        reqs[rid] = _make_fake_request(rid, [1, 2, 3, 100], [100])
    # After second accumulate, weights are [-2,-3,-4,-4].
    # ESS over all 4 (zombie-inclusive) ≈ 0.57 > 0.5 — no resample fired.
    ctrl.accumulate({"c0": -1.0, "c1": -1.0, "c2": -1.0})
    # Trigger freeze detection without resampling.
    ctrl.maybe_resample(reqs)

    weights = ctrl.get_final_weights("p1")
    # Slot 3 is frozen at the value it had when it was first detected as missing.
    assert weights[3] == pytest.approx(-4.0)
    # Slots 0-2 reflect current (post-accumulate) log_weights.
    # After the second accumulate, their weights are -2.0, -3.0, -4.0.
    # If resampling didn't fire (ESS >= 0.5), weights are still cumulative.
    assert weights[0] == pytest.approx(-2.0)
    assert weights[1] == pytest.approx(-3.0)
    assert weights[2] == pytest.approx(-4.0)


def test_get_final_weights_returns_empty_for_unknown_group() -> None:
    """get_final_weights() returns {} for an unregistered parent ID."""
    ctrl = SMCController()
    assert ctrl.get_final_weights("nonexistent") == {}


def test_zombie_inclusive_ess_trigger() -> None:
    """Resampling uses zombie-inclusive (all-N) ESS for the trigger.

    Sub-test 1: zombie weight better than active weights → zombie-inclusive
    ESS collapses below threshold → resample fires (zombie stays frozen,
    not included in loser list).

    Sub-test 2: all-N weights uniform → ESS = 1.0 → no resample.

    Sub-test 3: divergent active weights → ESS < threshold → resample fires.
    """
    # Sub-test 1: zombie has better weight (-1) than active weights (-4 each).
    # All-N ESS over [-4,-4,-4,-1] ≈ 0.33 < 0.5 → fires.
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
    )
    ctrl.accumulate({"c0": -1.0, "c1": -1.0, "c2": -1.0, "c3": -1.0})
    ctrl.accumulate({"c0": -3.0, "c1": -3.0, "c2": -3.0})
    # All-N weights: [-4, -4, -4, -1]; zombie (-1) dominates active (-4 each).
    reqs: dict[str, object] = {}
    for rid in ["c0", "c1", "c2"]:
        reqs[rid] = _make_fake_request(rid, [1, 2, 3, 100], [100])
    actions = ctrl.maybe_resample(reqs)
    assert "p1" in actions, (
        "Zombie-inclusive ESS collapse should trigger resample "
        "(zombie weight -1 dominates active -4)"
    )
    # c3 is a zombie — it must NOT appear in loser_request_ids.
    assert "c3" not in actions["p1"].loser_request_ids

    # Sub-test 2: all-N weights uniform → ESS = 1.0 → no resample.
    ctrl2 = SMCController()
    ctrl2.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
    )
    # All four accumulate at -1.0; c3 finishes → frozen at -1.0; active also -1.0.
    ctrl2.accumulate({"c0": -1.0, "c1": -1.0, "c2": -1.0, "c3": -1.0})
    # All-N weights: [-1, -1, -1, -1] → uniform → ESS = 1.0 → no resample.
    reqs2: dict[str, object] = {}
    for rid in ["c0", "c1", "c2"]:
        reqs2[rid] = _make_fake_request(rid, [1, 2, 3, 100], [100])
    actions2 = ctrl2.maybe_resample(reqs2)
    assert "p1" not in actions2, "Uniform all-N weights should not trigger resample"

    # Sub-test 3: divergent active weights DO trigger resample.
    ctrl3 = SMCController()
    ctrl3.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
    )
    ctrl3.accumulate({"c0": 0.0, "c1": -100.0, "c2": -100.0, "c3": -1.0})
    # c3 zombie (weight -1), active c0=0 dominates c1=-100, c2=-100.
    # All-N ESS ≈ 0.41 < 0.5 → fires.
    reqs3: dict[str, object] = {}
    for rid in ["c0", "c1", "c2"]:
        reqs3[rid] = _make_fake_request(rid, [1, 2, 3, 100], [100])
    actions3 = ctrl3.maybe_resample(reqs3)
    assert "p1" in actions3, "Divergent active weights should trigger resample"


def test_zombie_ancestor_does_not_win_active_slots() -> None:
    """Resampling pool is active-only: zombie ancestors cannot replace active
    slots.  Active losers always get live NewParticle replacements from active
    winners, never ZombieClone entries.  zombie_clones is always empty."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
        original_prompt_len=3,
    )
    # c0 finishes with weight 0 (best zombie); c1 has a good active weight,
    # c2/c3 have poor active weights → c1 should win among active slots.
    ctrl.accumulate({"c0": 0.0, "c1": -0.1, "c2": -100.0, "c3": -100.0})
    zombie_tokens = [1, 2, 3, 10, 11, 12]  # prompt + 3 generated tokens
    snapshots = {"c0": zombie_tokens}
    # c0 is zombie (absent from requests).
    reqs: dict[str, object] = {
        "c1": _make_fake_request("c1", [1, 2, 3, 100, 101], [100, 101]),
        "c2": _make_fake_request("c2", [1, 2, 3, 200, 201], [200, 201]),
        "c3": _make_fake_request("c3", [1, 2, 3, 300, 301], [300, 301]),
    }
    actions = ctrl.maybe_resample(reqs, token_snapshots=snapshots)

    assert "p1" in actions
    action = actions["p1"]
    # c0's zombie_token_ids should be populated from snapshots.
    group = ctrl._groups["p1"]
    assert group.zombie_token_ids.get(0) == zombie_tokens

    # No zombie clones — active-only resampling pool.
    assert action.zombie_clones == []

    # c2, c3 are active losers; each gets a NewParticle from active winner c1.
    assert set(action.loser_request_ids) == {"c2", "c3"}
    assert len(action.new_particles) == 2
    for p in action.new_particles:
        assert isinstance(p, NewParticle)
        assert p.ancestor_request_id == "c1"

    # After resample: active weights (c1, c2, c3 → slots 1, 2, 3) reset to 0.
    # Zombie slot 0 (c0) frozen_weight preserved (0.0 from before finishing).
    assert group.log_weights[1:] == [0.0, 0.0, 0.0]
    assert 0 in group.frozen_weights


def test_weights_reset_active_only_after_resample_with_zombie() -> None:
    """After resampling with a zombie present, only ACTIVE weights reset to 0.
    Zombie frozen_weights are preserved for get_final_weights() voting."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1", "c2", "c3"], alpha=2.0, ess_threshold=0.5,
        alpha_ramp_tokens=0, original_max_tokens=100,
    )
    # c3 finishes at -1.0; active c0 dominates with 0.0.
    ctrl.accumulate({"c0": 0.0, "c1": -100.0, "c2": -100.0, "c3": -1.0})
    ctrl.accumulate({"c0": -2.0, "c1": -2.0, "c2": -2.0})
    # Active weights: [-2, -102, -102]; c0 dominates active ESS → fires.
    reqs: dict[str, object] = {}
    for rid in ["c0", "c1", "c2"]:
        reqs[rid] = _make_fake_request(rid, [1, 2, 3, 100], [100])
    ctrl.maybe_resample(reqs)
    group = ctrl._groups["p1"]
    # Only active slots 0, 1, 2 reset to 0; zombie slot 3 frozen at -1.0.
    assert group.log_weights[:3] == [0.0, 0.0, 0.0]
    assert group.log_weights[3] == pytest.approx(-1.0)
    assert group.frozen_weights == {3: pytest.approx(-1.0)}


def test_exhausted_budget_skipped() -> None:
    """Particles whose ancestor exhausted max_tokens are not created."""
    ctrl = SMCController()
    ctrl.register_group(
        "p1", ["c0", "c1"], alpha=2.0, ess_threshold=0.99,
        alpha_ramp_tokens=0, original_max_tokens=5,
    )
    ctrl.accumulate({"c0": 0.0, "c1": -100.0})
    # Ancestor c0 has 5 output tokens = original_max_tokens → remaining=0
    reqs: dict[str, object] = {
        "c0": _make_fake_request("c0", [1, 2, 3, 100, 101, 102, 103, 104],
                                 [100, 101, 102, 103, 104]),
        "c1": _make_fake_request("c1", [1, 2, 3, 200, 201, 202, 203, 204],
                                 [200, 201, 202, 203, 204]),
    }
    actions = ctrl.maybe_resample(reqs)
    action = actions["p1"]
    # Slot 1 should have no new particle (remaining=0)
    assert len(action.new_particles) == 0


# ─── Phase 3: static output field checks (no model required) ─────────────────


def test_engine_core_output_has_smc_winner_token_ids_field() -> None:
    """EngineCoreOutput msgspec struct exposes smc_winner_token_ids."""
    eco_fields = {f.name for f in msgspec.structs.fields(EngineCoreOutput)}
    assert "smc_winner_token_ids" in eco_fields, (
        f"smc_winner_token_ids missing; fields={eco_fields}"
    )


def test_completion_output_has_smc_log_weight_field() -> None:
    """CompletionOutput dataclass exposes smc_log_weight."""
    field_names = {f.name for f in dataclass_fields(CompletionOutput)}
    assert "smc_log_weight" in field_names, (
        f"smc_log_weight missing; fields={field_names}"
    )


def test_engine_core_output_has_smc_log_weight_field() -> None:
    """EngineCoreOutput msgspec struct exposes smc_log_weight."""
    eco_fields = {f.name for f in msgspec.structs.fields(EngineCoreOutput)}
    assert "smc_log_weight" in eco_fields, (
        f"smc_log_weight missing; fields={eco_fields}"
    )


def test_engine_core_output_has_smc_detokenizer_reset_field() -> None:
    """EngineCoreOutput msgspec struct exposes smc_detokenizer_reset."""
    eco_fields = {f.name for f in msgspec.structs.fields(EngineCoreOutput)}
    assert "smc_detokenizer_reset" in eco_fields, (
        f"smc_detokenizer_reset missing; fields={eco_fields}"
    )


def test_completion_response_choice_has_smc_log_weight_field() -> None:
    """CompletionResponseChoice pydantic model exposes smc_log_weight."""
    from vllm.entrypoints.openai.completion.protocol import CompletionResponseChoice

    assert "smc_log_weight" in CompletionResponseChoice.model_fields, (
        f"smc_log_weight missing; fields={set(CompletionResponseChoice.model_fields)}"
    )
