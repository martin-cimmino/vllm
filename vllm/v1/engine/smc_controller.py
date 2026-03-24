# SPDX-License-Identifier: Apache-2.0
"""SMC controller: accumulates per-token weights and resamples particles."""
import math
import random
from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)


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

    loser_request_ids: list[str]  # IDs to abort (active + revived frozen)
    new_particles: list[NewParticle]  # replacement live requests to create


@dataclass
class ParticleGroup:
    """State for one SMC request group (one parent, N children)."""

    parent_request_id: str
    child_request_ids: list[str]
    log_weights: list[float]
    prefix_logprob: list[float]
    step_count: int = 0
    ess_threshold: float = 0.5
    alpha: float = 2.0
    alpha_ramp_tokens: int = 0
    original_max_tokens: int = 0  # user-specified max_tokens (for adjustments)
    original_prompt_len: int = 0  # length of original prompt tokens
    frozen_particles: set = field(default_factory=set)


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
            prefix_logprob=[0.0] * n,
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

    def accumulate(
        self,
        smc_log_weight_update: dict[str, float],
        sampled_logprob: dict[str, float],
        alpha_diff: dict[str, float],
    ) -> None:
        """Add incremental log-weights from this step to each group.
        
        The inputs are the values required to perform the log-weight update according to 
        Algorithm 1 and sec 5.3. of the SMC paper:
        - smc_log_weight_update: the incremental log-weight update for each particle 
          (line 10 in Algorithm 1 of the paper)
        - sampled_logprob: the log-probability of the newly sampled token for each 
          particle (sec. 5.3 of the paper)
        - alpha_diff: the difference in alpha values for this step vs. the previous step 
          (sec. 5.3 of the paper)
        """
        for pid, group in self._groups.items():
            for i, req_id in enumerate(group.child_request_ids):
                # only accumulate for active particles
                if req_id in smc_log_weight_update and i not in group.frozen_particles:
                    # accumulate logprob of the prefix p(y_1:t | x) according to sec 5.3. of paper
                    group.prefix_logprob[i] += sampled_logprob[req_id]
                    # perform log-weight update according to line 10 in Algorithm 1 of paper
                    group.log_weights[i] += smc_log_weight_update[req_id]
                    # perform log-weight update according to sec 5.3. of paper
                    group.log_weights[i] += alpha_diff[req_id] * group.prefix_logprob[i]
            group.step_count += 1

    @staticmethod
    def compute_ess(log_weights: list[float]) -> float:
        """Effective sample size, normalized to [0, 1].
        
        Given the weights w (which need not be normalized), we compute normalized 
        weights first
        w_norm_i = w_i / sum(w), 
        and then compute ESS as
        ESS = sum(w)^2 / (sum(w^2) * N)
        where N is the total number of weights.
        The input and computations are done in log-space for numerical stability.
        """
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

        u = random.uniform(0, 1.0)
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

        Terminology
        -----------
        A *frozen* particle is one that has finished generating (absent from
        ``requests``).  Its accumulated log-weight is stored in
        ``frozen_weights`` and never changes.

        ESS trigger
        -----------
        ESS is computed over **all N weights** (frozen-inclusive).  As active
        particles accumulate negative incremental weights while frozen weights
        stay fixed, a high-weight frozen particle can dominate → ESS collapses
        → resampling fires.

        Resampling pool
        ---------------
        Resampling is drawn from **all N slots** (frozen + active) via
        systematic resampling over ``group.log_weights``.  Three outcomes
        depending on the winning ancestor's status:

        * **Active ancestor wins** — the loser is replaced by a new vLLM
          request that clones the winner's token sequence.

          - Active loser: its live request is added to ``loser_request_ids``
            for abortion; the slot gets a fresh ``NewParticle``.
          - Frozen loser: it is removed from ``frozen_weights`` ("revived");
            the slot also gets a fresh ``NewParticle``.

        * **Frozen ancestor wins** — no new vLLM request is created (there is
          no live KV state to clone from).  The loser slot inherits the
          winner's frozen weight and is added to ``frozen_weights``.  If the
          loser was active, its live request is **not** aborted (to maintain
          the scheduler's N-slot invariant — aborting without replacement
          causes engine halt).  The slot is frozen and its future weight
          accumulation is suppressed via the ``accumulate()`` guard; the
          orphaned request runs to natural completion.

        * **Active ancestor with exhausted budget** — the ancestor has no
          remaining tokens.  No replacement can be created.  The loser slot
          is frozen in place (same treatment as frozen-ancestor wins).

        After resampling, only slots not present in ``frozen_weights`` have
        their ``log_weights`` reset to 0 (i.e. the uniform post-resample
        distribution).  Frozen slots retain their stored weight so that
        ``get_final_weights()`` can use them for importance-weighted voting.

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

            # 1. Detect newly finished particles; freeze their weight.
            for i, rid in enumerate(group.child_request_ids):
                if rid not in requests and i not in group.frozen_particles:
                    group.frozen_particles.add(i)

            # 2. Skip if all particles are finished.
            has_active = any(i not in group.frozen_particles for i in range(n))
            if not has_active:
                continue

            # 3. Check ESS over all N weights (frozen-inclusive).
            ess = self.compute_ess(group.log_weights)
            if ess >= group.ess_threshold:
                continue

            logger.debug("Log weights before resampling: %s",
                         group.log_weights)
            logger.debug("Frozen weights before resampling: %s",
                         group.frozen_particles)

            # 4. Systematic resample over all N slots (frozen + active).
            ancestors = self.systematic_resample(group.log_weights)
            ancestors = _rearrange_ancestors(ancestors)

            # 5. Gather token sequences from active ancestors (for cloning).
            ancestor_token_seqs: dict[int, list[int]] = {}
            ancestor_output_lens: dict[int, int] = {}
            ancestor_max_tokens: dict[int, int] = {}
            for slot in set(ancestors):
                if slot in group.frozen_particles:
                    continue  # Frozen ancestor — no live KV state to clone.
                rid = group.child_request_ids[slot]
                req = requests.get(rid)
                if req is not None:
                    ancestor_token_seqs[slot] = list(req.all_token_ids)  # type: ignore[union-attr]
                    ancestor_output_lens[slot] = len(req._output_token_ids)  # type: ignore[union-attr]
                    sp = req.sampling_params  # type: ignore[union-attr]
                    ancestor_max_tokens[slot] = (
                        sp.max_tokens if sp is not None
                        else group.original_max_tokens
                    )

            # 6. Build NewParticles for loser slots.
            # Proxy ancestors are valid because new requests are created before
            # losers are aborted (abort-after-create in core.py).
            loser_req_ids: list[str] = []
            new_particles: list[NewParticle] = []
            new_child_ids: list[str] = list(group.child_request_ids)
            frozen_weights_to_remove: list[int] = []

            # Snapshot which particles are frozen before the loop so that all
            # slot decisions are based on pre-resample state.  Without this,
            # a slot k that becomes frozen during iteration (because its own
            # ancestor is frozen) would cause later slots pointing to k to
            # incorrectly inherit its frozen weight instead of getting a new
            # particle from k.
            frozen_before_resample = group.frozen_particles.copy()

            for slot_idx in range(n):
                anc_idx = ancestors[slot_idx]
                if slot_idx == anc_idx:
                    continue  # Winner — stays in its slot.
                    
                # Store the prefix log prob of the ancestor in the group.
                group.prefix_logprob[slot_idx] = group.prefix_logprob[anc_idx]

                # --- Frozen ancestor: cannot clone (no live KV state). ---
                # Loser inherits the ancestor's frozen weight and becomes
                # frozen itself.  Active losers are NOT aborted (no
                # replacement → scheduler halt).
                if anc_idx in frozen_before_resample:
                    group.frozen_particles.add(slot_idx)
                    logger.debug(
                        "Slot %d: frozen ancestor %d (w=%.4f), "
                        "freezing slot", slot_idx, anc_idx,
                        group.frozen_particles)
                    continue

                # --- Active ancestor: check remaining token budget. ---
                token_ids = ancestor_token_seqs.get(anc_idx, [])
                num_output = ancestor_output_lens.get(anc_idx, 0)
                anc_max_tokens = ancestor_max_tokens.get(
                    anc_idx, group.original_max_tokens
                )
                remaining = anc_max_tokens - num_output
                if remaining <= 0:
                    # Ancestor exhausted — freeze slot, don't abort
                    # (same rationale as frozen-ancestor case).
                    group.frozen_particles.add(slot_idx)
                    logger.debug(
                        "Slot %d: ancestor %d exhausted budget "
                        "(max=%d, output=%d), freezing slot",
                        slot_idx, anc_idx, anc_max_tokens, num_output)
                    continue

                # --- Frozen loser revived by active ancestor: de-freeze
                # the slot and create a replacement particle.  The
                # output processor retains SMC child request states on
                # finish (smc_retained=True), so the replacement's
                # remapped output will find the RequestState alive. ---
                if slot_idx in frozen_before_resample:
                    frozen_weights_to_remove.append(slot_idx)
                    logger.debug(
                        "Slot %d: frozen loser revived by active "
                        "ancestor %d, de-freezing and creating "
                        "replacement", slot_idx, anc_idx)

                # --- Active or revived loser: create replacement. ---
                loser_req_ids.append(
                    group.child_request_ids[slot_idx]
                )

                new_id = f"smc_{pid}_{group.step_count}_{slot_idx}"
                original_child = self.get_original_id(
                    group.child_request_ids[slot_idx]
                )
                self._id_to_original[new_id] = original_child
                new_child_ids[slot_idx] = new_id

                new_particles.append(NewParticle(
                    new_request_id=new_id,
                    ancestor_request_id=group.child_request_ids[anc_idx],
                    token_ids=token_ids,
                    slot_index=slot_idx,
                    original_max_tokens=anc_max_tokens,
                    num_output_tokens=num_output,
                ))

            # De-freeze slots revived by an active winner.
            for slot_idx in frozen_weights_to_remove:
                group.frozen_particles.remove(slot_idx)

            logger.debug("Loser request IDs: %s", loser_req_ids)

            resample_actions[pid] = ResampleAction(
                loser_request_ids=loser_req_ids,
                new_particles=new_particles,
            )
            logger.debug("Old child IDs: %s", group.child_request_ids)
            logger.debug("New child IDs: %s", new_child_ids)

            group.child_request_ids = new_child_ids

            # 7. Reset all log-weights to 0.0 (uniform after resample).
            group.log_weights = [0.0] * len(group.log_weights)
            logger.debug("Log weights after resampling: %s",
                         group.log_weights)
            logger.debug("Frozen weights after resampling: %s",
                         group.frozen_particles)

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


def _rearrange_ancestors(ancestors: list[int]) -> list[int]:
    """Rearrange ancestors so that, if a particle is among the ancestors, it becomes 
    its own ancestor (i.e. wins itself and keeps its slot).
    
    This makes resampling more efficient without compromising its correctness.

    For example:
        input ancestors = [2, 0, 2, 1]
        output ancestors = [0, 1, 2, 2] (0, 1 and 2 become self ancestors)
    """
    unique_anc = set(ancestors)
    ancestors = ancestors.copy()
    for i in range(len(ancestors)):
        if i in unique_anc:
            # Find the next index of i in ancestors and swap it with the current
            # index. This has quadratic complexity, but we don't expect to have a
            # huge number of particles.
            idx = ancestors.index(i)
            ancestors[i], ancestors[idx] = ancestors[idx], ancestors[i]
    return ancestors