# SPDX-License-Identifier: Apache-2.0
"""SMC controller: accumulates per-token weights and resamples particles."""
import math
import random
from dataclasses import dataclass, field


@dataclass
class NewParticle:
    """Describes a new request to create for a resampled slot."""

    new_request_id: str
    ancestor_request_id: str  # winner whose tokens to clone
    token_ids: list[int]  # winner's all_token_ids (prompt + generated)
    slot_index: int  # position in the particle group
    original_max_tokens: int  # original user-specified max_tokens
    num_output_tokens: int  # how many output tokens the ancestor has generated


@dataclass
class ZombieClone:
    """A zombie particle clone — no vLLM request, just controller tracking."""

    new_request_id: str
    slot_index: int
    token_ids: list[int]  # copied from ancestor zombie (or active→zombie)


@dataclass
class ResampleAction:
    """Structured output from a resampling event."""

    loser_request_ids: list[str]  # active-slot IDs to abort
    new_particles: list[NewParticle]  # replacement live requests to create
    zombie_clones: list[ZombieClone] = field(default_factory=list)


@dataclass
class ParticleGroup:
    """State for one SMC request group (one parent, N children)."""

    parent_request_id: str
    child_request_ids: list[str]
    log_weights: list[float]
    step_count: int = 0
    ess_threshold: float = 0.5
    alpha: float = 2.0
    alpha_ramp_tokens: int = 0
    original_max_tokens: int = 0  # user-specified max_tokens (for adjustments)
    original_prompt_len: int = 0  # length of original prompt tokens
    # slot_index → cumulative log-weight frozen when that particle finished.
    # Cleared after each resampling (zombies re-freeze next step with weight 0).
    frozen_weights: dict = field(default_factory=dict)
    # slot_index → token_ids of finished particle (preserved across resampling).
    zombie_token_ids: dict = field(default_factory=dict)


class SMCController:
    """Accumulates per-token SMC log-weights and triggers resampling."""

    def __init__(self) -> None:
        self._groups: dict[str, ParticleGroup] = {}
        # Maps internal SMC request IDs → original external-facing child IDs.
        # Chains through multiple resampling events.
        self._id_to_original: dict[str, str] = {}

    def register_group(
        self,
        parent_request_id: str,
        child_request_ids: list[str],
        alpha: float,
        ess_threshold: float,
        alpha_ramp_tokens: int,
        original_max_tokens: int = 0,
        original_prompt_len: int = 0,
    ) -> None:
        n = len(child_request_ids)
        self._groups[parent_request_id] = ParticleGroup(
            parent_request_id=parent_request_id,
            child_request_ids=list(child_request_ids),
            log_weights=[0.0] * n,
            ess_threshold=ess_threshold,
            alpha=alpha,
            alpha_ramp_tokens=alpha_ramp_tokens,
            original_max_tokens=original_max_tokens,
            original_prompt_len=original_prompt_len,
        )

    def unregister_group(self, parent_request_id: str) -> None:
        self._groups.pop(parent_request_id, None)

    def get_original_id(self, request_id: str) -> str:
        """Resolve an internal SMC request ID to its original child ID.

        If not mapped (i.e. it's already the original), returns itself.
        """
        return self._id_to_original.get(request_id, request_id)

    def accumulate(self, smc_log_weights: dict[str, float]) -> None:
        """Add incremental log-weights from this step to each group."""
        for pid, group in self._groups.items():
            for i, req_id in enumerate(group.child_request_ids):
                if req_id in smc_log_weights:
                    group.log_weights[i] += smc_log_weights[req_id]
            group.step_count += 1
            #if group.step_count % 200 == 0:
                #active = sum(1 for i in range(len(group.child_request_ids))
                #             if i not in group.frozen_weights)
                #print(
                #    f"[SMC_HB] group={pid} step={group.step_count} "
                #    f"active={active}/{len(group.child_request_ids)} "
                #    f"frozen={len(group.frozen_weights)}",
                #    flush=True,
                #)

    @staticmethod
    def compute_ess(log_weights: list[float]) -> float:
        """Effective sample size, normalized to [0, 1]."""
        n = len(log_weights)
        if n == 0:
            return 0.0
        max_lw = max(log_weights)
        w = [math.exp(lw - max_lw) for lw in log_weights]
        sum_w = sum(w)
        if sum_w == 0.0:
            return 0.0
        sum_w2 = sum(wi * wi for wi in w)
        return (sum_w * sum_w) / (sum_w2 * n) if sum_w2 > 0 else 0.0

    @staticmethod
    def systematic_resample(log_weights: list[float]) -> list[int]:
        """Return ancestor indices via systematic resampling."""
        n = len(log_weights)
        max_lw = max(log_weights)
        # Subtract max_lw for numerical stability (doesn't change relative weights).
        w = [math.exp(lw - max_lw) for lw in log_weights]
        cumsum: list[float] = []
        # Comulative sum of weights for systematic resampling.
        s = 0.0
        for wi in w:
            s += wi
            cumsum.append(s)
        total = cumsum[-1]
        # random offset in [0, total/n) for systematic resampling (even spacing with random start).
        u = random.uniform(0, 1.0 / n)
        ancestors: list[int] = []
        j = 0
        # Iterate over evenly spaced positions in [0, total) and find their ancestor slots.
        for i in range(n):
            target = (u + i) / n * total
            while j < n - 1 and cumsum[j] < target:
                j += 1
            ancestors.append(j)
        return ancestors

    def maybe_resample(
        self,
        requests: dict[str, "object"],
        token_snapshots: "dict[str, list[int]] | None" = None,
    ) -> dict[str, ResampleAction]:
        """Check ESS for each group; resample if below threshold.

        ESS is computed over ALL N weights (zombie-inclusive): finished
        particles keep their frozen weight while active particles accumulate
        negative incremental weights, so zombie weights rapidly dominate →
        ESS collapses → resampling fires frequently (matching reference).

        Resampling itself operates over ACTIVE slots only.  Zombies cannot
        produce new vLLM requests (they have no KV state to resume from), so
        they are excluded from the resampling pool.  Zombie weights contribute
        to the ESS trigger but winners are always drawn from active particles.

        After resampling ALL N weights are reset to 0 (uniform). Zombie slots
        re-freeze at 0 on the next step.

        Args:
            requests: scheduler's requests dict (str → Request objects).
            token_snapshots: optional rid → all_token_ids from last active step,
                used to populate zombie_token_ids for newly finished particles.

        Returns:
            Mapping parent_req_id → ResampleAction for groups that resampled.
        """
        resample_actions: dict[str, ResampleAction] = {}
        snaps = token_snapshots or {}

        for pid, group in list(self._groups.items()):
            n = len(group.child_request_ids)
            if n == 0:
                continue

            # Step 1: Detect newly finished particles; freeze weight + snapshot.
            for i, rid in enumerate(group.child_request_ids):
                if rid not in requests and i not in group.frozen_weights:
                    group.frozen_weights[i] = group.log_weights[i]
                    snap = snaps.get(rid)
                    if snap is not None:
                        group.zombie_token_ids[i] = snap

            # Step 2: Identify active (non-frozen) slots and their weights.
            active_list = sorted(
                i for i in range(n) if i not in group.frozen_weights
            )
            if not active_list:
                continue  # All particles finished — nothing to resample.

            # Step 3: ESS over ACTIVE weights only.
            #
            # NOTE: zombie-inclusive ESS is NOT used here, despite the
            # reference power_smc.py doing so.  The reason is architectural:
            # after each resample all weights reset to 0; zombies re-freeze
            # at 0 the next step while active particles accumulate negative
            # incremental weights.  Zombie weights (0) always dominate active
            # weights (negative) → zombie-inclusive ESS collapses on EVERY
            # step → resampling fires every step → active particles are
            # perpetually aborted before finishing → permanent hang.
            #
            # The reference avoids this because zombie ancestors can win slots
            # and produce done=True clones, shrinking the active pool until
            # done.all().  vLLM cannot replicate that (finished requests have
            # no live KV state), so active-only ESS is the correct trigger.
            active_weights = [group.log_weights[i] for i in active_list]
            ess = self.compute_ess(active_weights)
            if ess >= group.ess_threshold:
                continue

            #print(
            #    f"Resampling group {pid} at step {group.step_count} "
            #    f"(ESS={ess:.3f} < {group.ess_threshold}, "
            #    f"active={len(active_list)}/{n})", flush=True)

            # Step 4: Systematic resample over ACTIVE slots only.
            anc_pos = self.systematic_resample(active_weights)
            # anc_pos[j] → position in active_list that is the ancestor of
            # active_list[j].  active_list[anc_pos[j]] is the ancestor slot.

            # Step 5: Identify TRUE winners (self-mapped positions in anc_pos)
            # and resolve proxy ancestors.
            #
            # Systematic resampling can produce "proxy ancestors": a slot k
            # appears in anc_pos (some position maps to k) but k itself is a
            # loser (anc_pos[k] != k, so k maps to some other slot l).  Using
            # k as ancestor_request_id is wrong because k will be aborted,
            # causing the replacement to be silently skipped.  The fix is to
            # resolve every position's ancestry chain to the nearest TRUE winner
            # (a self-mapped position), then use only true winners as ancestors.
            true_winner_pos: set[int] = {
                j for j in range(len(active_list)) if anc_pos[j] == j
            }

            def _resolve(pos: int) -> int:
                """Follow anc_pos chain until a self-mapped (true winner) pos."""
                seen: set[int] = set()
                while pos not in true_winner_pos:
                    if pos in seen:
                        break  # cycle guard (shouldn't happen with valid resample)
                    seen.add(pos)
                    pos = anc_pos[pos]
                return pos

            # Collect token sequences only for TRUE winner slots.
            true_winner_slots = {active_list[j] for j in true_winner_pos}
            winner_token_seqs: dict[int, list[int]] = {}
            winner_output_lens: dict[int, int] = {}
            winner_max_tokens: dict[int, int] = {}
            for slot in true_winner_slots:
                rid = group.child_request_ids[slot]
                req = requests.get(rid)
                if req is not None:
                    winner_token_seqs[slot] = list(req.all_token_ids)  # type: ignore[union-attr]
                    winner_output_lens[slot] = len(req._output_token_ids)  # type: ignore[union-attr]
                    sp = req.sampling_params  # type: ignore[union-attr]
                    winner_max_tokens[slot] = (
                        sp.max_tokens if sp is not None
                        else group.original_max_tokens
                    )

            # Step 6: Build NewParticles for loser active slots.
            # Use resolved true-winner ancestors (not proxy intermediaries).
            active_loser_req_ids: list[str] = []
            new_particles: list[NewParticle] = []
            new_child_ids: list[str] = list(group.child_request_ids)

            for j, slot_idx in enumerate(active_list):
                true_anc_j = _resolve(j)
                true_anc_slot = active_list[true_anc_j]
                if slot_idx == true_anc_slot:
                    continue  # True winner — stays in its slot.

                active_loser_req_ids.append(group.child_request_ids[slot_idx])

                new_id = f"smc_{pid}_{group.step_count}_{slot_idx}"
                original_child = self.get_original_id(
                    group.child_request_ids[slot_idx]
                )
                self._id_to_original[new_id] = original_child
                new_child_ids[slot_idx] = new_id

                token_ids = winner_token_seqs.get(true_anc_slot, [])
                num_output = winner_output_lens.get(true_anc_slot, 0)
                anc_max_tokens = winner_max_tokens.get(
                    true_anc_slot, group.original_max_tokens
                )
                remaining = anc_max_tokens - num_output
                if remaining <= 0:
                    continue
                new_particles.append(NewParticle(
                    new_request_id=new_id,
                    ancestor_request_id=group.child_request_ids[true_anc_slot],
                    token_ids=token_ids,
                    slot_index=slot_idx,
                    original_max_tokens=anc_max_tokens,
                    num_output_tokens=num_output,
                ))

            #print(
            #    f"[SMC_DIAG] resample pid={pid} step={group.step_count} "
            #    f"active={len(active_list)} losers={len(active_loser_req_ids)} "
            #    f"new_particles={len(new_particles)} "
            #    f"true_winners={sorted(true_winner_slots)} "
            #    f"loser_ids={active_loser_req_ids[:4]}{'...' if len(active_loser_req_ids)>4 else ''}",
            #    flush=True,
            #)
            #for i, p in enumerate(new_particles):
                #print(
                #    f"[SMC_DIAG]   particle[{i}] new_id={p.new_request_id} "
                #    f"anc={p.ancestor_request_id} "
                #    f"token_ids_len={len(p.token_ids)} "
                #    f"num_output={p.num_output_tokens} "
                #    f"remaining={p.original_max_tokens - p.num_output_tokens}",
                #    flush=True,
                #)

            resample_actions[pid] = ResampleAction(
                loser_request_ids=active_loser_req_ids,
                new_particles=new_particles,
                zombie_clones=[],
            )
            group.child_request_ids = new_child_ids

            # Step 7: Reset ACTIVE weights to 0 (uniform after resample).
            # Zombie frozen_weights are preserved — they represent accumulated
            # SMC weight at finishing time and are needed for get_final_weights().
            # Clearing them would reset zombie weights to 0, which would cause
            # zombie-inclusive ESS to collapse every subsequent step (infinite loop).
            for i in active_list:
                group.log_weights[i] = 0.0

        return resample_actions

    def get_final_weights(self, parent_request_id: str) -> dict:
        """Return per-slot final weights: frozen weight for finished particles,
        current accumulated weight for still-active particles.

        Returns a dict mapping slot_index → log_weight.
        """
        group = self._groups.get(parent_request_id)
        if group is None:
            return {}
        result: dict = {}
        for i in range(len(group.child_request_ids)):
            if i in group.frozen_weights:
                result[i] = group.frozen_weights[i]
            else:
                result[i] = group.log_weights[i]
        return result
