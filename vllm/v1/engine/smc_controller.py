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
class ResampleAction:
    """Structured output from a resampling event."""

    loser_request_ids: list[str]  # active-slot IDs to abort
    new_particles: list[NewParticle]  # replacement live requests to create


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
    # Preserved across resampling events; zombie weights stay frozen for
    # get_final_weights() voting.
    frozen_weights: dict = field(default_factory=dict)


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

        After resampling, active slot weights are reset to 0 (uniform). Zombie
        frozen_weights are preserved so that get_final_weights() can use them.

        Args:
            requests: scheduler's requests dict (str → Request objects).

        Returns:
            Mapping parent_req_id → ResampleAction for groups that resampled.
        """
        resample_actions: dict[str, ResampleAction] = {}

        for pid, group in list(self._groups.items()):
            n = len(group.child_request_ids)
            if n == 0:
                continue

            # Step 1: Detect newly finished particles; freeze weight.
            for i, rid in enumerate(group.child_request_ids):
                if rid not in requests and i not in group.frozen_weights:
                    group.frozen_weights[i] = group.log_weights[i]

            # Step 2: Identify active (non-frozen) slots and their weights.
            active_list = sorted(
                i for i in range(n) if i not in group.frozen_weights
            )
            if not active_list:
                continue  # All particles finished — nothing to resample.

            # Step 3: ESS over ACTIVE weights only.
            active_weights = [group.log_weights[i] for i in active_list]
            #ess = self.compute_ess(active_weights)
            ess = self.compute_ess(group.log_weights)
            
            if ess >= group.ess_threshold:
                continue

            # Step 4: Systematic resample over ACTIVE slots only as we cannot clone zombies.
            anc_pos = self.systematic_resample(active_weights)

            # Get ancestor slots (indices in child_request_ids) for each active slot.
            ancestor_slots = {active_list[j] for j in anc_pos}
            #print(f"active_list={active_list} anc_pos={anc_pos} ancestor_slots={ancestor_slots} -> {[active_list[j] for j in anc_pos]} winners: {[active_list[anc_pos[i]] for i in range(len(anc_pos)) if active_list[anc_pos[i]]==active_list[i]]}")
            
            ancestor_token_seqs: dict[int, list[int]] = {}
            ancestor_output_lens: dict[int, int] = {}
            ancestor_max_tokens: dict[int, int] = {}
            for slot in ancestor_slots:
                rid = group.child_request_ids[slot]
                req = requests.get(rid)
                if req is not None:
                    #print(f"Found active ancestor request {rid} with tokens {len(list(req.all_token_ids))}")
                    ancestor_token_seqs[slot] = list(req.all_token_ids)  # type: ignore[union-attr]
                    ancestor_output_lens[slot] = len(req._output_token_ids)  # type: ignore[union-attr]
                    sp = req.sampling_params  # type: ignore[union-attr]
                    ancestor_max_tokens[slot] = (
                        sp.max_tokens if sp is not None
                        else group.original_max_tokens
                    )

            # Step 6: Build NewParticles for loser active slots.
            # Use resolved true-winner ancestors (not proxy intermediaries).
            active_loser_req_ids: list[str] = []
            new_particles: list[NewParticle] = []
            new_child_ids: list[str] = list(group.child_request_ids)

            for j, slot_idx in enumerate(active_list):
                true_anc_slot = active_list[anc_pos[j]]
                if slot_idx == true_anc_slot:
                    continue  # True winner — stays in its slot.

                #print(f"j={j} slot_idx={slot_idx} anc_pos={anc_pos[j]} true_anc_slot={true_anc_slot}")

                active_loser_req_ids.append(group.child_request_ids[slot_idx])

                new_id = f"smc_{pid}_{group.step_count}_{slot_idx}"
                original_child = self.get_original_id(
                    group.child_request_ids[slot_idx]
                )
                self._id_to_original[new_id] = original_child
                new_child_ids[slot_idx] = new_id

                token_ids = ancestor_token_seqs.get(true_anc_slot, [])
                num_output = ancestor_output_lens.get(true_anc_slot, 0)
                anc_max_tokens = ancestor_max_tokens.get(
                    true_anc_slot, group.original_max_tokens
                )
                remaining = anc_max_tokens - num_output
                if remaining <= 0:
                    print(f"Ancestor slot {true_anc_slot} has no remaining tokens (max={anc_max_tokens} output={num_output}), skipping new particle")
                    continue
                
                new_particles.append(NewParticle(
                    new_request_id=new_id,
                    ancestor_request_id=group.child_request_ids[true_anc_slot],
                    token_ids=token_ids,
                    slot_index=slot_idx,
                    original_max_tokens=anc_max_tokens,
                    num_output_tokens=num_output,
                ))

            #print(f"Resampling group {pid}: {len(active_loser_req_ids)} losers")
            #print(f"Loser request IDs: {active_loser_req_ids}")

            resample_actions[pid] = ResampleAction(
                loser_request_ids=active_loser_req_ids,
                new_particles=new_particles,
            )
            #print("Old child IDs:", group.child_request_ids)
            #print("New child IDs after resampling:", new_child_ids)

            # update group state for new particles: replace losers with new IDs, keep winners in place.
            group.child_request_ids = new_child_ids

            #breakpoint()

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
