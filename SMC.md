# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a **fork of vLLM** being extended to support **Power-SMC (Sequential Monte Carlo) sampling** for LLM inference. The goal is to implement per-token SMC resampling natively inside vLLM's engine for efficient particle-based decoding.

The model used for experiments: `/leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16`

The reference HF SMC implementation lives at: `/leonardo_scratch/fast/AIFAC_L13_018/mcimmino/power-smc`

Current branch: `power-smc`

---

## Environment (Leonardo HPC Cluster)

### Setup

The venv lives at `/leonardo_scratch/fast/AIFAC_L13_018/mcimmino/vllm/.venv`. Activate before running anything:

```bash
cd /leonardo_scratch/fast/AIFAC_L13_018/mcimmino/vllm
source /leonardo_scratch/fast/AIFAC_L13_018/mcimmino/vllm/.venv/bin/activate
module load cuda/12.2
```

**Dev tools** (`ruff`, `pytest`, `tblib`) are **not** installed by the default vLLM install — add separately if missing:
```bash
uv pip install ruff pytest pytest-asyncio tblib
```

If re-installing vLLM (Python-only changes — no C++ modifications):
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

Batch job (2h):
```bash
sbatch <script>.sbatch -- --arg1 val1
```

### Smoke test
```bash
python smoke_test.py --model /leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16 --prompt "Hello world" --max_tokens 128
```

---

## Linting and Tests

```bash
# Lint / format
ruff check vllm/
ruff format vllm/

# Type checking
mypy vllm/

# Run a single test file
pytest tests/v1/sample/test_sampler.py -x -v
```

### Power-SMC Test Suite

| File | Phase | GPU required |
|---|---|---|
| `tests/v1/sample/test_smc_sampler.py` | 1a — `Sampler._compute_smc_weights` math | Yes |
| `tests/v1/sample/test_smc_sampling_params.py` | 1b — `SamplingParams` SMC validation | No |
| `tests/v1/engine/test_smc_controller.py` | 2a — `SMCController` unit + Phase 3 field checks | No |
| `tests/v1/engine/test_smc_e2e.py` | 2b+3 — integration + `smc_log_weight` access | Yes |

```bash
# No-GPU suite:
pytest tests/v1/sample/test_smc_sampling_params.py tests/v1/engine/test_smc_controller.py -v

# GPU tests (on a compute node):
export SMC_TEST_MODEL=/leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16
pytest tests/v1/sample/test_smc_sampler.py tests/v1/engine/test_smc_e2e.py -v

# All SMC tests:
pytest tests/v1/sample/test_smc_*.py tests/v1/engine/test_smc_*.py -v
```

**Last known status (2026-03-18):**
- No-GPU suite (44 tests): **all pass**.
  - `test_smc_sampling_params.py` (11): all pass.
  - `test_smc_controller.py` (33): all pass — covers ESS, systematic resampling, register/accumulate/unregister lifecycle, `ResampleAction` construction, winner preservation, ID chaining, `max_tokens` adjustment, budget exhaustion, zombie-inclusive ESS trigger (`test_zombie_inclusive_ess_trigger`), zombie-exclusive resampling pool (`test_zombie_ancestor_does_not_win_active_slots`), active-only weight reset (`test_weights_reset_active_only_after_resample_with_zombie`), proxy-ancestor assertions (`test_maybe_resample_losers_identified_correctly`), and lifecycle/Phase 3 field tests.
- GPU sampler suite (`test_smc_sampler.py`): not re-run after Phase 4 alpha-ramp fix (minor).
- GPU e2e suite (`test_smc_e2e.py`): not re-run after bug fixes — awaiting GPU node. Set `SMC_TEST_MODEL` env var.
- `smc_benchmark_v4.py`: validated 2026-03-13 (pre-zombie-ESS and pre-abort-after-create fix). **Re-run needed** to validate current state.
- `ruff check vllm/`: clean on all SMC-modified files.

---

## Architecture: V1 Engine (Critical)

vLLM has two engine versions. **This project targets V1 exclusively.**

| Concept | V0 | V1 |
|---|---|---|
| Parallel sampling | `SequenceGroup` with N `Sequence` objects, explicit `fork()` | N independent `Request` objects via `ParentRequest` |
| KV sharing | Explicit CoW `fork(parent, child)` | Prefix caching via `BlockPool` ref counts |
| Resampling primitive | `BlockManager.fork()` + `free()` | Create replacement requests, then abort losers (prefix cache reuses winner KV blocks) |

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

### SMC Data Flow (V1)

```
EngineCore.step()
  → schedule()
  → execute_model()
      Sampler: logsumexp(α * log_softmax(logits)) per SMC request
      → SamplerOutput.smc_log_weights: dict[req_id, float]
  → SMCController.accumulate(smc_log_weights)
  → update _smc_token_snapshot for all active SMC particles
  → SMCController.maybe_resample(scheduler.requests, token_snapshots)
      freeze newly finished slots (zombie); save token_ids to zombie_token_ids
      compute ESS over ALL N weights (zombie-inclusive)
      if ESS < threshold AND active > 0:
        → systematic resample over ACTIVE slots only
        → each loser slot → NewParticle with direct ancestor (proxy OK)
        → return ResampleAction(loser_request_ids, new_particles)
        → reset ACTIVE weights to 0; zombie frozen_weights preserved
  → EngineCore._apply_resample_actions(actions)
      → Request(prompt=ancestor.all_token_ids) → add_request()  # FIRST
        (prefix cache auto-hits ancestor's KV blocks — zero-copy reuse)
      → abort_requests(loser_ids)                                # THEN
        (abort-after-create: proxy ancestors still alive when cloned)
  → scheduler.update_from_output()
  → _smc_remap_outputs()
      → rewrite internal SMC IDs → original external child IDs
      → set smc_detokenizer_reset=True on first output of each replacement
  → OutputProcessor: clear detokenizer state on reset outputs
```

### Key Design Decisions

**Zombie-inclusive ESS trigger, active-only resampling pool:**
- ESS is computed over all N weights (including frozen zombie weights). As active particles accumulate negative incremental weights while zombie weights stay frozen, zombie weights dominate → ESS collapses → resampling fires.
- Resampling draws winners from **active slots only**. Zombies have no live KV state and cannot be resumed as new vLLM requests.
- After resample: active slot weights reset to 0; zombie `frozen_weights` preserved for `get_final_weights()` voting.

**Proxy ancestors (abort-after-create):**
`systematic_resample` can assign a slot k as ancestor for slot j, while k itself is also a loser. In `_apply_resample_actions()`, new requests are created **before** losers are aborted, so proxy ancestors are still alive in the scheduler when their `all_token_ids` are read. No chain-following needed in `smc_controller.py` — direct ancestors are always valid at creation time.

**Detokenizer reset (once per replacement):**
`smc_detokenizer_reset=True` is set only on the **first output** of each replacement particle (tracked via `_smc_new_particle_ids: set[str]`). Setting it on every output would clear the detokenizer each step, leaving only the last token in the final output.

### Child Request ID Convention

SMC child requests: `"{index}_{parent_id}"` — e.g. `"0_req-abc"`, `"1_req-abc"` (set by `ParallelSamplingProcessor.get_child_info()`).

Resampled replacement requests: `"smc_{parent_id}_{step_count}_{slot_index}"` — e.g. `"smc_req-abc_5_2"`.

**Important:** Child requests are created with `n=1`. Group registration happens **lazily on the first forward step** via `EngineCore._smc_auto_register_from_weights()`, which groups IDs in `smc_log_weights` by `parent_id`.

---

## Benchmark / Prototype Files

| File | Phase | Description |
|---|---|---|
| `smc_prototype_v2.py` | 0 | Batched offline inference, post-hoc logprob weights, no resampling. |
| `smc_benchmark_v3.py` | 1–3 | In-GPU engine weights vs post-hoc logprob weights. No live resampling. |
| `smc_benchmark_v4.py` | 2 | **Live resampling benchmark.** ESS trajectory, resampling events, voting accuracy. |
| `smc_benchmark_v5.py` | 4 | **AIME 2025 multi-problem benchmark.** Baseline vs SMC, aggregate metrics. |
| `run_smc_benchmark_v4.sbatch` | — | SLURM wrapper for `smc_benchmark_v4.py` (2h). |
| `run_smc_benchmark_v5.sbatch` | — | SLURM wrapper for `smc_benchmark_v5.py` (8h, 30 problems). |

### Running benchmarks

```bash
# smc_benchmark_v4 — single problem with baseline comparison:
python smc_benchmark_v4.py \
    --model /leonardo_scratch/fast/AIFAC_L13_018/models/Domyn-Small-v0.2-bf16 \
    --n_particles 16 --alpha 2.0 --ess_threshold 0.5 \
    --problem_idx 2 --compare_baseline

# smc_benchmark_v5 — debug run (3 problems, GPU node):
python smc_benchmark_v5.py --method both --max_problems 3 \
    --n_particles 4 --alpha 2.0 --ess_threshold 0.5 \
    --max_new_tokens 512 --seed 42

# smc_benchmark_v5 — full AIME 2025 (batch, 8h):
sbatch run_smc_benchmark_v5.sbatch -- --method both --n_particles 16 \
    --alpha 2.0 --ess_threshold 0.5 --max_new_tokens 8192

# Pre-download AIME dataset (login node, one-time):
python -c "from datasets import load_dataset; load_dataset('yentinglin/aime_2025', split='train'); print('OK')"
```

**What `smc_benchmark_v5.py` measures:**
- `avg@k`: mean(n_correct / k) ≈ expected accuracy of one random draw
- `pass@k`: fraction of problems where any sample is correct (oracle)
- `majority_vote_acc`: unweighted majority vote
- `weighted_majority_acc`: SMC weight-weighted vote
- `snis_draw_acc`: single SNIS draw from weight distribution
- ESS diagnostics: `mean_min_ess_x_n`, `total_resampling_events`

Output: `smc_results/smc_benchmark_v5_{method}_a{alpha}_n{n}_tau{tau}_s{seed}.json`

### Engine Weight vs Logprob Weight

| | Formula | What it measures |
|---|---|---|
| **Logprob weight** | `Σₜ log p(xₜ \| x<ₜ)` | Log-likelihood under base model |
| **Engine weight** | `Σₜ logsumexp(α · log_softmax(logits))` | Exact SMC importance weight; log-normalizer of `p̃ ∝ p^α` |

Engine weight ≤ logprob weight (more negative). Rank correlation typically >0.97.

### smc_benchmark_v4 Results (2026-03-13, pre-abort-after-create fix)

**Run:** n=16, α=3.0, τ=0.3, seed=42, model=Domyn-Small-v0.2-bf16, problem=nested quadratic (answer=-12)

| Metric | Baseline (τ=0.001) | Resampling (τ=0.3) |
|---|---|---|
| Steps | 2818 | 1762 |
| Resampling events | 0 | 12 |
| Mean ESS ×N | 1.36 | 5.75 |
| Unique answers | 2 | 11 |
| Majority vote | -12 ✓ | 2 ✗ |
| Weighted vote | -6 ✗ | -12 ✓ |
| Wall clock | 54.7s | 46.0s |

Notable: weighted vote correctly identifies the minority correct answer (-12) via SNIS. **Re-run needed** after zombie-inclusive ESS and abort-after-create fixes — expect more resampling events and no hang.

---

## Implementation Status

### Phases 1–3 — COMPLETE

- **Phase 1**: `SamplingParams` SMC fields; `Sampler._compute_smc_weights` in-GPU; `smc_log_weights` in `SamplerOutput`/`ModelRunnerOutput`; `SMCController` (accumulate, ESS, systematic resample).
- **Phase 2**: Mid-generation resampling wired end-to-end. `ResampleAction`/`NewParticle`/`ZombieClone` dataclasses; `_apply_resample_actions()` in `core.py`; ID remapping; detokenizer reset (once per replacement).
- **Phase 3**: `smc_log_weight` exposed through `CompletionOutput`, `EngineCoreOutput`, `CompletionResponseChoice`.

**Bugs fixed (all resolved):**
- `ValueError: SMC requires n > 1` during `SamplingParams.clone()`: removed `n<=1` guard from `_verify_args()` since child requests legitimately have `n=1` with `smc_alpha` set.
- Replacement particles producing only 1 output token: `smc_detokenizer_reset=True` was firing every decode step. Fixed via `_smc_new_particle_ids` set in `EngineCore`.
- Proxy-ancestor hang: `systematic_resample` assigns proxy ancestors (slots that are themselves losers). Fixed by creating new requests **before** aborting losers in `_apply_resample_actions()` (abort-after-create), so proxy ancestors are still alive when cloned. No chain-following needed.

### Phase 4 — IN PROGRESS (2026-03-18)

**Completed:**
- `smc_benchmark_v5.py` + SLURM wrapper: full AIME 2025 benchmark with aggregate metrics.
- Alpha ramp off-by-one fix in `sampler.py`: `(steps + 1.0)` instead of `steps`.
- Zombie-inclusive ESS trigger: `compute_ess(group.log_weights)` (all N) instead of active-only.
- Abort-after-create in `_apply_resample_actions()`: eliminates proxy-ancestor hang.
- `test_smc_controller.py` aligned with current behavior (33 tests, all pass): proxy-ancestor assertions in `test_maybe_resample_losers_identified_correctly`; renamed `test_zombie_inclusive_ess_trigger`.

**Pending:**
1. **Re-run `smc_benchmark_v4.py`** to validate all fixes end-to-end (zombie-inclusive ESS + abort-after-create). Expect more resampling events, all N particles finish cleanly, correct weighted vote.
2. **Run full AIME 2025 benchmark**: `sbatch run_smc_benchmark_v5.sbatch -- --method both --n_particles 16 --alpha 2.0 --ess_threshold 0.5`.
3. **Re-run GPU tests** (`test_smc_e2e.py`) on a compute node.

---

## Known Limitations and Open Issues

### 1. Streaming not supported
`smc_detokenizer_reset=True` clears the detokenizer — safe only in FINAL_ONLY mode. Streaming clients would receive corrupted output. Fix: reconstruct detokenizer state from winner's full token sequence before emitting deltas.

### 2. `_smc_new_particle_ids` cleanup on abort
When a replacement particle is aborted at the next resampling, its ID stays in `_smc_new_particle_ids` (it never had a first output). Currently harmless (stale IDs never fire again), but grows unboundedly over many resampling events.

### 3. ESS check uses strict `<`
ESS == threshold does not trigger resampling. With N=2, degenerate weights give ESS=0.5, which equals the default threshold and does NOT resample. Use `smc_ess_threshold=0.6` for N=2 groups.

### 4. `smc_detokenizer_reset` through ZMQ path untested
Only tested in single-process mode. The msgpack field (`omit_defaults=True`) should work correctly through EngineCoreProc ZMQ path but has not been validated.

### 5. Resampled requests have no priority inheritance
New requests start with `priority=0, status=WAITING`. Irrelevant for single-request SMC but could cause scheduling skew under load.

### 6. CUDA graph compatibility
`_apply_resample_actions()` calls `scheduler.add_request()` during the forward pass. May require disabling CUDA graphs for SMC requests.

---

## Linting Config

- **Ruff** for linting and formatting (`pyproject.toml`). Star imports (`F403`, `F405`) and lambda assignments (`E731`) are ignored.
- **mypy** with pydantic plugin for type checking.
