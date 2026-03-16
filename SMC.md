# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a **fork of vLLM** being extended to support **Power-SMC (Sequential Monte Carlo) sampling** for LLM inference. The goal is to implement per-token SMC resampling natively inside vLLM's engine for efficient particle-based decoding. See `PLAN.md` for the full architecture and implementation plan, and `RESULTS.md` for installation notes.

The model used for experiments: `/leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16`

The reference HF SMC implementation lives at: `/leonardo_scratch/fast/AIFAC_L13_018/mcimmino/power-smc`

Current branch: `power-smc`

---

## Environment (Leonardo HPC Cluster)

### Installation (already done — do not reinstall unless needed)

The venv lives at `/leonardo_scratch/fast/AIFAC_L13_018/mcimmino/vllm/.venv`. Activate before running anything:

```bash
cd /leonardo_scratch/fast/AIFAC_L13_018/mcimmino/vllm
source /leonardo_scratch/fast/AIFAC_L13_018/mcimmino/vllm/.venv/bin/activate
module load cuda/12.2
```

**Dev tools** (`ruff`, `pytest`, `tblib`) are **not** installed by the default vLLM install — they must be added separately. If missing:
```bash
uv pip install ruff pytest pytest-asyncio tblib
```

If re-installing vLLM is needed (Python-only changes only — no C++ modifications):
```bash
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
VLLM_PRECOMPILED_WHEEL_VARIANT=cu126 VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=cu126
uv pip install ruff pytest pytest-asyncio tblib datasets
```

Required env vars for jobs:
```bash
export HF_HOME="/leonardo_scratch/fast/AIFAC_L13_018/mcimmino/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SMC_TEST_MODEL=/leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16
```

### Running on the cluster

Interactive GPU session (for dev/debug, 30 min):
```bash
srun -N 1 --partition boost_usr_prod --nodes=1 --account=AIFAC_L13_018 --qos boost_qos_dbg --ntasks=1 --exclusive --time 00:30:00 --gres=gpu:1 --pty /bin/bash
```

Batch job (2h, for experiments):
```bash
sbatch vllm/run_smc_prototype_v2.sbatch -- --arg1 val1
```

### Smoke test (validate GPU + model loading)
```bash
python smoke_test.py --model /leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16 --prompt "Hello world" --max_tokens 128
```

### Run Phase 0 v2 prototype
```bash
python smc_prototype_v2.py --model /leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16 --n_particles 8 --alpha 2.0
```

---

## Linting and Tests

```bash
# Lint
ruff check vllm/
ruff format vllm/

# Type checking
mypy vllm/

# Run a single test
pytest tests/v1/sample/test_sampler.py -x -v

# Run V1 engine tests
pytest tests/v1/engine/ -x -v
```

### Power-SMC Test Suite

| File | Phase | GPU required |
|---|---|---|
| `tests/v1/sample/test_smc_sampler.py` | 1a — `Sampler._compute_smc_weights` math | Yes |
| `tests/v1/sample/test_smc_sampling_params.py` | 1b — `SamplingParams` SMC validation | No |
| `tests/v1/engine/test_smc_controller.py` | 2a — `SMCController` unit + Phase 3 field checks | No |
| `tests/v1/engine/test_smc_e2e.py` | 2b+3 — integration + `smc_log_weight` access | Yes |

```bash
# Unit tests only (no GPU needed):
pytest tests/v1/sample/test_smc_sampling_params.py tests/v1/engine/test_smc_controller.py -v

# GPU tests (run on a compute node — set SMC_TEST_MODEL to avoid HF download):
export SMC_TEST_MODEL=/leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16
pytest tests/v1/sample/test_smc_sampler.py tests/v1/engine/test_smc_e2e.py -v

# All SMC tests:
pytest tests/v1/sample/test_smc_*.py tests/v1/engine/test_smc_*.py -v
```

**Last known status (2026-03-15):**
- No-GPU suite (43 tests): **all pass**.
  - `test_smc_sampling_params.py` (11): all pass. Note: `n>1` validation removed from `SamplingParams._verify_args()` (child requests have `n=1` with `smc_alpha` set); no test relied on it.
  - `test_smc_controller.py` (32): all pass — includes Phase 2 tests for `ResampleAction`, winner preservation, ID chaining, `max_tokens` adjustment, budget exhaustion; zombie-inclusive ESS tests (`test_zombie_inclusive_ess_fires_more_frequently`, `test_zombie_clone_created_from_zombie_ancestor`, `test_weights_reset_all_n_after_resample_with_zombie`); and lifecycle tests (`test_freezes_finished_particle_and_resamples_active`, `test_all_particles_finished_skips_group`, `test_get_final_weights_merges_frozen_and_active`, `test_get_final_weights_returns_empty_for_unknown_group`).
- GPU sampler suite (`test_smc_sampler.py`): previously fixed device mismatch; not re-run after Phase 4 alpha-ramp fix (minor).
- GPU e2e suite (`test_smc_e2e.py`): not re-run after bug fixes (awaiting GPU node). Set `SMC_TEST_MODEL` env var to avoid HF download.
- `smc_benchmark_v4.py`: **end-to-end validated** (2026-03-13) — 12 resampling events (pre-zombie fix), correct weighted vote, meaningful per-particle token counts. Re-run needed after proxy-ancestor bug fix.
- `ruff check vllm/`: **clean** on all SMC-modified files (proxy-ancestor fix uses a nested `def _resolve`, which ruff accepts).

---

## Architecture: V1 Engine (Critical)

vLLM has two engine versions. **This project targets V1 exclusively.** V1 is architecturally different from V0:

| Concept | V0 | V1 |
|---|---|---|
| Parallel sampling | `SequenceGroup` with N `Sequence` objects, explicit `fork()` | N independent `Request` objects via `ParentRequest` |
| KV sharing | Explicit CoW `fork(parent, child)` | Prefix caching via `BlockPool` ref counts |
| Resampling primitive | `BlockManager.fork()` + `free()` | Abort loser requests + create new requests (prefix cache reuses KV blocks) |

### Key V1 Files for Power-SMC

| File | Role |
|---|---|
| `vllm/sampling_params.py` | `SamplingParams` — `smc_alpha`, `smc_ess_threshold`, `smc_alpha_ramp_tokens` |
| `vllm/v1/sample/sampler.py` | `Sampler._compute_smc_weights` — computes `logsumexp(α * log_softmax(logits))` in-GPU |
| `vllm/v1/outputs.py` | `SamplerOutput`, `ModelRunnerOutput` — `smc_log_weights` field |
| `vllm/v1/engine/smc_controller.py` | `SMCController` — accumulates weights, resamples, tracks ID remapping |
| `vllm/v1/engine/core.py` | `EngineCore` — SMC auto-registration, resampling wiring, output ID remapping |
| `vllm/v1/engine/__init__.py` | `EngineCoreOutput` — `smc_log_weight`, `smc_detokenizer_reset` fields |
| `vllm/v1/engine/output_processor.py` | Detokenizer reset on resampled outputs |
| `vllm/v1/request.py` | `Request`, `RequestStatus` |
| `vllm/outputs.py` | `CompletionOutput` — `smc_log_weight` field for API response |
| `vllm/entrypoints/openai/completion/protocol.py` | `CompletionResponseChoice` — `smc_log_weight` field |

### SMC Data Flow (V1, Phase 4)

```
EngineCore.step()
  → schedule()
  → execute_model()
      Sampler: logsumexp(α * log_softmax(logits)) per SMC request
      → SamplerOutput.smc_log_weights: dict[req_id, float]
  → SMCController.accumulate(smc_log_weights)
  → update _smc_token_snapshot for all active SMC particles
      (particles that finish this step still in scheduler.requests here)
  → SMCController.maybe_resample(scheduler.requests, token_snapshots)
      freeze newly finished slots; save token_ids to zombie_token_ids
      compute ESS over ALL N weights (zombie-inclusive)
      if ESS < threshold AND active > 0:
        → systematic resample over all N slots
        → active ancestor → NewParticle (live replacement request)
        → zombie ancestor → ZombieClone (controller tracking only)
        → return ResampleAction(loser_request_ids, new_particles, zombie_clones)
        → reset ALL N weights to 0; clear frozen_weights
  → EngineCore._apply_resample_actions(actions)
      → abort_requests(active_loser_ids)     # free loser KV blocks
      → Request(prompt=winner.all_token_ids) → add_request()  # per NewParticle
        (prefix cache auto-hits winner blocks — zero-copy KV reuse)
      → ZombieClones: no new request; controller state already updated
  → scheduler.update_from_output()
  → _smc_remap_outputs()
      → rewrite internal SMC IDs → original external child IDs
      → set smc_detokenizer_reset=True on remapped outputs
  → OutputProcessor: clear detokenizer state on reset outputs
```

### Child Request ID Convention

SMC child requests follow the pattern `"{index}_{parent_id}"` — e.g. `"0_req-abc"`, `"1_req-abc"`, etc. (set by `ParallelSamplingProcessor.get_child_info()`).

Resampled (replacement) requests use: `"smc_{parent_id}_{step_count}_{slot_index}"` — e.g. `"smc_req-abc_5_2"`.

**Important:** Child requests are created with `child_sampling_params.n = 1` (one generation per child). The SMC group cannot be detected by checking `sp.n` from a child request — it will always be 1. Group registration therefore happens **lazily on the first forward step** via `EngineCore._smc_auto_register_from_weights()`, which reads all active request IDs from `smc_log_weights`, groups them by `parent_id`, and registers one group per parent.

### Benchmark / Prototype Files

| File | Phase | Description |
|---|---|---|
| `smc_prototype.py` | 0 v1 | Token-by-token, slow. Historical reference only. |
| `smc_prototype_v2.py` | 0 v2 | Batched offline inference, post-hoc logprob weights, no resampling. |
| `smc_benchmark_v3.py` | 1–3 | In-GPU engine weights vs post-hoc logprob weights. No live resampling. |
| `smc_benchmark_v4.py` | 2 | **Live resampling benchmark.** Runs with and without resampling, tracks ESS trajectory, resampling events, voting accuracy. |
| `smc_benchmark_v5.py` | 4 | **AIME 2025 multi-problem benchmark.** Baseline vs SMC across all 30 problems; computes aggregate `avg@k`, `pass@k`, `majority_vote_acc`, `weighted_majority_acc`, `snis_draw_acc`. |
| `run_smc_prototype_v2.sbatch` | — | SLURM wrapper for `smc_prototype_v2.py`. |
| `run_smc_benchmark_v4.sbatch` | — | SLURM wrapper for `smc_benchmark_v4.py`. |
| `run_smc_benchmark_v5.sbatch` | — | SLURM wrapper for `smc_benchmark_v5.py` (8h budget for 30 problems). |

#### Running `smc_benchmark_v4.py`

```bash
# Single problem, with comparison against no-resampling baseline:
python smc_benchmark_v4.py \
    --model /leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16 \
    --n_particles 8 --alpha 2.0 --ess_threshold 0.5 \
    --problem_idx 2 --compare_baseline

# All problems, resampling only:
python smc_benchmark_v4.py --all_problems --n_particles 8 --alpha 2.0 --ess_threshold 0.5

# Via SLURM (2h batch job):
sbatch run_smc_benchmark_v4.sbatch -- --n_particles 16 --ess_threshold 0.5 --compare_baseline
```

#### Running `smc_benchmark_v5.py` (AIME 2025, 30 problems)

```bash
# Pre-download dataset (login node, one-time):
python -c "from datasets import load_dataset; load_dataset('yentinglin/aime_2025', split='train'); print('OK')"

# Debug run — 3 problems, both methods (interactive GPU node):
python smc_benchmark_v5.py --method both --max_problems 3 \
    --n_particles 4 --alpha 2.0 --ess_threshold 0.5 \
    --max_new_tokens 512 --seed 42

# Full AIME 2025 run — all 30 problems (batch, 8h):
sbatch run_smc_benchmark_v5.sbatch -- --method both --n_particles 16 \
    --alpha 2.0 --ess_threshold 0.5 --max_new_tokens 8192
```

**What `smc_benchmark_v5.py` measures (aggregate over 30 problems):**
- **`avg@k`**: mean(n_correct / k) across problems ≈ expected accuracy of one random draw (pass@1)
- **`pass@k`**: fraction of problems where any sample is correct (oracle)
- **`majority_vote_acc`**: pass@1 using unweighted majority vote (baseline and SMC)
- **`weighted_majority_acc`**: pass@1 using cumulative SMC weight-weighted vote (SMC only)
- **`snis_draw_acc`**: pass@1 using a single SNIS draw from the weight distribution (SMC only)
- **ESS diagnostics**: `mean_min_ess_x_n`, `total_resampling_events`

**JSON output schema:**
```json
{
  "summary": {
    "config": {"n_particles": 16, "alpha": 2.0, "ess_threshold": 0.5, ...},
    "baseline": {"avg_at_k": 0.1, "pass_at_k": 0.6, "majority_vote_acc": 0.4, ...},
    "smc": {"majority_vote_acc": 0.4, "weighted_majority_acc": 0.5, "snis_draw_acc": 0.45, ...}
  },
  "details": [{"problem_idx": 0, "problem": {...}, "runs": {"baseline": {...}, "smc": {...}}}, ...]
}
```

Output filename: `smc_results/smc_benchmark_v5_{method}_a{alpha}_n{n}_tau{tau}_s{seed}.json`

**What `smc_benchmark_v4.py` measures:**
- **ESS trajectory**: min/mean/final ESS × N across all generation steps (should stay above τ×N with resampling, collapses to ~1 without)
- **Resampling events**: step index, loser IDs, winner slots, remaining budget for each event
- **Voting accuracy**: majority vote and cumulative-weight-weighted vote vs ground truth
- **Answer diversity**: number of unique final answers
- **Wall-clock overhead**: generation time with vs without resampling

**Key difference from v3:** v3 captures weights but never resamples (all particles run to completion). v4 has live resampling active: particles that fall behind are replaced mid-generation by copies of winners, which reuse winner KV blocks via prefix cache.

### Engine Weight vs Logprob Weight

Two weight types are tracked in the benchmark:

| | Formula | What it measures |
|---|---|---|
| **Logprob weight** | `Σₜ log p(xₜ \| x<ₜ)` | Log-likelihood of the sequence under the base model; returned directly by vLLM's `logprob` output |
| **Engine weight** | `Σₜ logsumexp(α · log_softmax(logits))` | Exact SMC importance weight; log-normalizer of the twisted distribution `p̃ ∝ p^α` |

The engine weight is always ≤ logprob weight (i.e., more negative). The gap at each step equals the entropy-weighted penalty from the twist: high-entropy (flat distribution) steps contribute a larger gap; near-deterministic steps contribute near-zero gap. High rank correlation between the two (typically >0.97) confirms consistency.

### smc_benchmark_v4 Results (2026-03-13)

**Run:** n=16, α=3.0, τ=0.3, seed=42, max_new_tokens=8192, model=Domyn-Small-v0.2-bf16
**Problem:** Algebra — nested quadratic (ground truth = -12)
**Result file:** `smc_results/smc_benchmark_v4_a3.0_n16_tau0.3_s42_cmp.json`

| Metric | Baseline (τ=0.001) | Resampling (τ=0.3) |
|---|---|---|
| Steps (tokens/N) | 2818 | 1762 |
| Resampling events | 0 | 12 |
| Mean ESS ×N | 1.36 | 5.75 |
| Unique answers | 2 | 11 |
| Majority vote | -12 ✓ | 2 ✗ |
| Weighted vote | -6 ✗ | -12 ✓ |
| Wall clock | 54.7s | 46.0s |

Key observations:

1. **Resampling is working end-to-end.** Particles now have meaningful token counts (52–1103 tokens). Mean ESS 5.75 vs 1.36 baseline. 12 resampling events spread across the full generation.

2. **Weighted vote correct, majority wrong.** Only 2/16 particles answered -12 (particles 0 and 15), but they have the best weights (-137, -118). SNIS correctly identifies the good paths even as a minority — this validates the weight computation.

3. **Diversity dramatically improved.** 11 unique answers (resampling) vs 2 (baseline). Resampling forces exploration of diverse paths.

4. **Speed improvement from resampling.** 46.0s vs 54.7s (-16%), 1762 vs 2818 steps (-37.5%). Prefix-cache KV reuse on cloned particles reduces total token generation. Early convergence on good paths terminates the run sooner.

5. **α=3.0 causes very rapid weight collapse.** First resampling at step 14 (degeneracy in just 14 tokens). Higher α makes the twisted distribution more peaked, so particles diverge faster. Min ESS still hits 1.0×N (complete degeneracy) despite τ=0.3 — the collapse outpaces the check frequency. α=2.0 with τ=0.5 likely gives better balance.

6. **Length bias is severe.** Particle 9 (1103 tokens, weight −752) vs particle 15 (193 tokens, weight −118). Both wrong, but particle 9's weight is 6× more negative purely from length. Per-token normalization would significantly reshape ESS and voting.

7. **Particle 0 (correct) was itself resampled away at step 146.** It became a loser and was replaced by a clone — the 590-token correct output is from a later `smc_` replacement of the same slot. ID-remapping chain is working correctly.

8. **Clustering of resampling events.** Events at steps 784–833 and 1549–1567 fire in rapid succession, suggesting particles collapse quickly after each resampling under strong α=3.0 twist. Adaptive α ramp (α_ramp_tokens) could help here.

### smc_benchmark_v3 Results (2026-03-13)

**Run:** n=16, α=2.0, τ=0.5, seed=42, model=Domyn-Small-v0.2-bf16
**Problem:** Algebra — nested quadratic (ground truth = -12)
**Result file:** `smc_results/smc_benchmark_v3_a2.0_n16_s42.json`

Key observations:

1. **ESS ≈ 1.0/16 — complete weight degeneracy.** Without mid-generation resampling, accumulated log-weights over hundreds of steps collapse to a point mass on one particle. This is the canonical SMC weight degeneracy problem and is exactly what Phase 2 resampling is designed to prevent.

2. **Majority vote correct, weighted vote wrong.** 8/16 particles answered -12 (correct), 8/16 answered -6. Majority vote picked -12 ✓. All weighted estimators (engine SNIS, logprob SNIS) picked -6 ✗, because the dominant particle (particle 15, 748 tokens, weight -30) gave -6.

3. **Length bias in weights.** Shorter sequences accumulate less negative weight regardless of correctness. Per-token average weight would partially correct this.

4. **Rank correlation = 0.979.** Engine and logprob weights agree on ranking — the exact vs approximate weight distinction doesn't matter much for voting, but the exact engine weight is needed for principled resampling.

5. **One particle (idx 9) hit max_steps** without finishing — its weight is among the worst, no effect on voting.

---

## Implementation Phases

### Phase 1 — COMPLETE (2026-03-12)
- Extended `SamplingParams` with `smc_alpha`, `smc_ess_threshold`, `smc_alpha_ramp_tokens`.
- Modified `Sampler._compute_smc_weights` to compute `logsumexp(α * log_softmax(logits))` in-GPU per request.
- Added `smc_log_weights: dict[str, float]` to `SamplerOutput` and `ModelRunnerOutput`.
- `SMCController`: accumulate/ESS/systematic-resample logic.
- `CompletionOutput.smc_log_weight`, `EngineCoreOutput.smc_log_weight`, `CompletionResponseChoice.smc_log_weight` fields added.

### Phase 2 — COMPLETE (2026-03-13)
Mid-generation SMC resampling wired end-to-end.

**Changes:**
- `vllm/v1/engine/smc_controller.py`: Added `ResampleAction`, `NewParticle`, and `ZombieClone` dataclasses. `maybe_resample()` accepts `requests: dict` and returns structured `ResampleAction` objects. Added `_id_to_original` dict for ID remapping across chained resampling events. Winners stay in their slot (no-op); only loser slots get new IDs.
- `vllm/v1/engine/core.py`:
  - `_smc_pending_groups`: accumulates child requests until all N arrive, then calls `smc_controller.register_group()`.
  - `_smc_maybe_register()`: auto-registration logic, detects `"{index}_{parent_id}"` ID pattern.
  - `_apply_resample_actions()`: aborts losers, creates new `Request` objects from winner `all_token_ids` (with adjusted `max_tokens`), adds to scheduler. Prefix cache auto-reuses winner KV blocks.
  - `_smc_remap_outputs()`: rewrites internal SMC IDs → original external child IDs in `EngineCoreOutputs`; sets `smc_detokenizer_reset=True` **only on the first output** of each replacement particle (tracked via `_smc_new_particle_ids` set). This is critical — setting it on every output would clear the detokenizer each step, leaving only 1 token in the final output.
  - Both `step()` and `step_with_batch_queue()` wired with real resampling logic (placeholders removed).
- `vllm/v1/engine/__init__.py`: Added `smc_detokenizer_reset: bool = False` to `EngineCoreOutput`.
- `vllm/v1/engine/output_processor.py`: Clears detokenizer `token_ids` when `smc_detokenizer_reset=True`, so winner's continuation is detokenized cleanly (FINAL_ONLY mode).

**Bugs fixed post-Phase 2 (2026-03-13):**
- `ValueError: SMC requires n > 1` on `SamplingParams.clone()` during resampling: child requests legitimately have `n=1` with `smc_alpha` set; removed the `n<=1` guard from `_verify_args()`.
- **Replacement particles producing only 1 output token**: `smc_detokenizer_reset=True` was being set on every output from a remapped particle. The detokenizer was cleared every decode step, so only the last token survived. Fixed by tracking which replacement particle IDs have not yet had their first output fired (`_smc_new_particle_ids: set[str]` in `EngineCore`), setting the reset flag exactly once, then clearing the ID from the set.

**Bugs fixed post-Phase 4 (2026-03-15):**
- **Proxy-ancestor hang** (`smc_controller.py`): `systematic_resample` can return ancestor assignments where a slot k appears as the ancestor for some other slot j (`anc_pos[j] = k`), while k itself is also a loser (`anc_pos[k] != k` — k maps to some other true winner l). The previous code used slot k's request ID as `ancestor_request_id` for the new particle replacing j. But k is in `loser_request_ids` and gets aborted immediately before the new-particle loop runs, so `scheduler.requests.get(k_id)` returns None → the replacement is silently skipped. Losers with skipped replacements are never delivered an output → the parent request waits forever → **hang**.
  - **Fix** (`smc_controller.py` `maybe_resample`): compute `true_winner_pos` (positions where `anc_pos[j] == j`, i.e. self-mapped), then for every loser j, call `_resolve(anc_pos[j])` to follow the chain until reaching a true-winner position. Use the true-winner's slot as `ancestor_request_id` and token sequence. Collect token sequences only from true-winner slots. This eliminates all `ancestor_not_found` skips.
  - **Root cause confirmed** with `[SMC_DBG] SKIP reason=ancestor_not_found in_sched=False` diagnostics: at step 27, 14 losers were built but 8 replacements were silently dropped because their direct ancestors (proxy slots 4,6,7,8,9,10,12,13) were themselves losers and already aborted.

**Design decisions:**
- FINAL_ONLY mode for Phase 2 MVP: streaming support deferred to Phase 4. Mid-stream detokenizer reset would break streaming clients.
- No scheduler changes needed: new requests enter through normal `add_request()` path; prefix cache handles KV reuse automatically.
- No explicit fork/CoW: V1's hash-based prefix cache provides zero-copy KV reuse when the new request's `prompt_token_ids` equals the winner's `all_token_ids`.

### Phase 3 — COMPLETE (concurrent with Phase 1/2)
- `smc_log_weight` exposed through `CompletionOutput`, `EngineCoreOutput`, and `CompletionResponseChoice`.
- `smc_detokenizer_reset` added to `EngineCoreOutput` (Phase 2).

### Phase 4 — IN PROGRESS (2026-03-15)
Multi-problem benchmark on AIME 2025, reference alignment, and production hardening.

**Completed:**
- `smc_benchmark_v5.py` + `run_smc_benchmark_v5.sbatch`: full 30-problem AIME 2025 benchmark comparing baseline vs Power-SMC with aggregate metrics (`avg@k`, `pass@k`, `majority_vote_acc`, `weighted_majority_acc`, `snis_draw_acc`).
- Dataset: `yentinglin/aime_2025` loaded via HF `datasets` (cached at `HF_HOME`).
- Both v4 and v5 benchmarks use `SMCController.get_final_weights()` for weighted voting; `random.seed(seed)` called in `run_one()`/`run_smc()` for deterministic resampling trajectories.
- **Reference alignment investigation (2026-03-15)**: Compared with `power_smc.py`; adopted correct vLLM-compatible design. See Issue #8 for full analysis.
  - `SMCController.maybe_resample()` uses **active-only ESS** for the trigger and **active-only resampling pool**. Zombie-inclusive ESS cannot be used (causes infinite resampling loop — see Issue #8).
  - Zombie `frozen_weights` are preserved across resampling events (not cleared) so that zombie accumulated weights survive for `get_final_weights()` voting.
  - `ParticleGroup.zombie_token_ids` and `_smc_token_snapshot` retained for potential future use.
  - Alpha ramp off-by-one fix in `sampler.py`: `(steps + 1.0)` instead of `steps` — ramp starts at alpha_t > 1 at step 0, matching reference.

**Completed (continued, 2026-03-15):**
- **Proxy-ancestor hang fixed**: `SMCController.maybe_resample()` now resolves every loser's ancestry chain to the true winner (self-mapped position) before using as `ancestor_request_id`. All replacements now create successfully; no more silent skips; no hang.

**Pending:**
- Re-run `smc_benchmark_v4.py` to validate end-to-end after proxy-ancestor fix (expect all N particles to finish cleanly, correct weighted vote).
- Run full AIME 2025 benchmark (`smc_benchmark_v5.py`).
- `test_smc_controller.py` likely needs new tests for the proxy-ancestor case (weights that trigger a proxy in `systematic_resample`).

---

## Known Limitations and Open Issues (Phase 2+)

### 1. ~~GPU E2E test not yet run after Phase 2~~ — Validated by benchmark (2026-03-13), re-validation pending after proxy-ancestor fix
`smc_benchmark_v4.py` ran successfully end-to-end with live resampling (12 events, correct weighted vote, meaningful token counts per particle). Three critical bugs found and fixed:
- `ValueError: SMC requires n > 1` during `SamplingParams.clone()` on child request.
- Replacement particles producing 1 output token due to `smc_detokenizer_reset=True` firing on every decode step instead of only the first.
- **Proxy-ancestor hang** (2026-03-15): systematic resampling can produce proxy ancestors (slots used as ancestors that are themselves losers), causing all replacements pointing to them to be silently skipped → hang. Fixed in `maybe_resample` by resolving ancestry chains to true winners.
Re-run `smc_benchmark_v4.py` to validate all fixes together. Formal `test_smc_e2e.py` still needed for prefix cache hit rate.

### 2. Streaming not supported
When `smc_detokenizer_reset=True`, the detokenizer is cleared. This is safe only in FINAL_ONLY mode. Streaming clients would receive corrupted output (winner tokens appended to loser's partial text). Fix for Phase 4: instead of clearing, reconstruct detokenizer state from winner's full token sequence before emitting deltas.

### 3. OutputProcessor doesn't know about resampled IDs
The ID remapping in `_smc_remap_outputs()` rewrites `EngineCoreOutput.request_id` before `OutputProcessor.process_outputs()` is called. This means the OutputProcessor sees the original child ID and finds the correct `RequestState`. However, mid-stream outputs from the *loser* (before resampling) have already been emitted with the original child ID — in FINAL_ONLY mode this is harmless, but in streaming mode it would result in a mix of loser and winner tokens.

### 4. ~~`_smc_pending_groups` never cleaned up on abort~~
Resolved: `_smc_pending_groups` was removed entirely. Group registration is now lazy (from first step's `smc_log_weights`) so there is no partial-registration state to clean up.

### 5. Resampled request is always WAITING (no priority inheritance)
New requests created in `_apply_resample_actions()` start with default `priority=0` and `status=WAITING`. If the scheduler deprioritizes long-waiting requests, resampled particles might be delayed relative to surviving particles. Currently not an issue for single-request SMC, but could matter under high load.

### 6. `smc_detokenizer_reset` not propagated through EngineCoreProc ZMQ path
`EngineCoreOutput.smc_detokenizer_reset` is a `bool` field with `omit_defaults=True`. When `False` (the default), it's omitted from the msgpack-encoded message — correct. When `True` (after resampling), it's included and decoded on the frontend. This should work, but has not been tested end-to-end through the ZMQ multiprocessing path (only single-process mode tested).

### 7. ESS check uses strict `<` — ESS exactly equal to threshold does not resample
`compute_ess(lw) >= group.ess_threshold` means ESS == threshold is treated as "no resample". This is intentional (threshold is a minimum acceptable diversity floor), but worth noting: with N=2, degenerate weights give ESS = 0.5, which equals the default `smc_ess_threshold=0.5` and does NOT trigger resampling. Use `smc_ess_threshold=0.6` or higher for N=2 groups.

### 8. ~~Resampling stops once any particle finishes~~ — PARTIALLY RESOLVED (2026-03-15)

**Root cause:** The reference HF implementation (`power_smc.py`) includes finished ("zombie") particles in both the ESS computation and the resampling pool. Because active particles keep accumulating negative incremental weights while zombie weights stay frozen (incremental masked to 0), zombie weights rapidly dominate → ESS collapses → resampling fires hundreds of times per run.

**Reference implementation behaviour (`power_smc.py` lines 211–231):**
- ESS computed over all N weights (including `done` particles).
- `systematic_resample(log_weights)` draws ancestors over **all N slots**.
- `done = done[ancestors]` — zombie ancestor → new slot inherits `done=True`; on the next step, that slot's incremental weight is forced to 0 and it generates EOS.
- All N weights reset to 0 after resample.
- Condition: `ess < kappa * N and not done.all()`.

**Previous vLLM approach (Option E):** ESS computed over active-only weights; only active slots resampled; frozen weights preserved across resamplings. This gave too few resampling events (~12 vs reference ~655) because active-only ESS stays high.

**Current vLLM approach — active-only ESS trigger AND active-only pool** (finalised 2026-03-15):
- `maybe_resample()` computes ESS over **active weights only**; triggers if `ESS < threshold AND active > 0`. ✗ differs from reference trigger sensitivity.
- Resamples over **active slots only** — zombie slots excluded from the pool. ✗ differs from reference.
- Losers among active slots get `NewParticle` replacements from active winners. `zombie_clones` is always empty.
- After resample: **active weights reset to 0**; zombie `frozen_weights` preserved.
- `SMCController.get_final_weights(parent_id)` returns frozen weight for finished slots and current `log_weight` for active ones — use for final weighted voting.

**Why NOT zombie-inclusive ESS trigger (despite the reference using it):**
Zombie-inclusive ESS + weight reset causes an infinite resampling loop in vLLM:
1. After any resample, zombie frozen_weights must be cleared so zombies re-freeze at 0 (consistent with "all weights uniform after resample")
2. Next step: zombies stay at weight 0, active particles accumulate small negative incremental weights
3. Zombie weights (0) dominate active weights (negative) → zombie-inclusive ESS collapses EVERY step
4. Resampling fires every step → active particles perpetually aborted before finishing → permanent hang

The reference avoids this because zombie ancestors win slots and produce `done=True` clones, so the active pool shrinks toward zero and `done.all()` terminates generation. vLLM cannot replicate this (finished requests have no live KV state), so active-only ESS is the only safe trigger.

**Why NOT all-N resampling pool:**
When a zombie is selected as an ancestor in the reference, the new slot inherits `done=True` and immediately stops generating (weight 0). Replicating this requires synthesizing a finished output and delivering it through vLLM's output pipeline without causing an engine deadlock — which proved non-trivial. Active-only pool avoids this entirely.

**Behavioural comparison:**

| Scenario | Current vLLM | Reference (`power_smc.py`) |
|---|---|---|
| ESS computed from | Active weights only | All N weights |
| Zombie effect on ESS trigger | None | Dominates → fires every step (after first zombie) |
| Resampling pool | Active only | All N slots |
| Zombie ancestor | Not eligible | Clones `done=True` slot |
| Weight reset after resample | Active slots → 0.0; zombie frozen_weights preserved | ALL N slots → 0.0 |
| Group terminates when | ALL slots frozen | `done.all()` |
| Zombie sequences dominate final ensemble | No | Yes (zombie clones fill slots) |
| Resampling events | ~12 (similar to original Option E) | ~hundreds |

**Practical consequence:** Our ensemble never concentrates purely on zombie (finished) sequences. Active particles always continue generating. Resampling fires when active particles' weights diverge — typically ~10–50 times per generation depending on α and model diversity. The reference's all-N pool concentrates probability mass on best-finished sequences but requires architectural support vLLM cannot provide.

---

## Next Steps (Phase 4)

### Immediate priorities

1. **Run AIME 2025 full benchmark** (`smc_benchmark_v5.py`): `sbatch run_smc_benchmark_v5.sbatch -- --method both --n_particles 16 --alpha 2.0 --ess_threshold 0.5`. Pre-download dataset on login node first.

2. **Formal `test_smc_e2e.py` re-run**: Run on a GPU node to verify prefix cache hit rate after resampling and check `smc_log_weight` propagation.

3. **Validate zombie-inclusive ESS trigger end-to-end** with `smc_benchmark_v4.py`:
   ```bash
   python smc_benchmark_v4.py \
       --model /leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16 \
       --n_particles 16 --alpha 2.0 --ess_threshold 0.5 \
       --problem_idx 2 --compare_baseline
   # Expect: more resampling events than the previous ~12 (zombie ESS trigger
   #         fires more often as zombie weights dominate); active-only pool means
   #         zombie sequences do not fill slots (unlike reference). No hang.
   ```

### Medium-term

4. **`_smc_new_particle_ids` cleanup on abort**: When a replacement particle is aborted (becomes a loser at the next resampling), its ID stays in `_smc_new_particle_ids` indefinitely (it never had a first output). Add cleanup in `abort_requests()` to discard aborted IDs from the set. Currently harmless (stale entries never fire since the ID is never seen again in outputs), but prevents unbounded growth over many resampling events.

5. **CUDA graph compatibility**: `_apply_resample_actions()` calls `scheduler.add_request()` during the forward pass. Verify this is safe with CUDA graphs enabled (likely requires disabling graphs for SMC requests, or deferring to post-step).

6. **Streaming support**: In `output_processor.py`, when `smc_detokenizer_reset=True` and the request is streaming, reconstruct detokenizer state from winner's tokens instead of clearing. Emit a "reset" chunk or suppress until next token.

7. **Multi-request SMC**: Test multiple simultaneous SMC parent requests. Each group is independent in `SMCController._groups`, so should work — but needs validation under load.

---

## Linting Config

- **Ruff** is used for linting and formatting (`pyproject.toml`).
- **mypy** with pydantic plugin for type checking.
- Star imports (`F403`, `F405`) and lambda assignments (`E731`) are ignored.
