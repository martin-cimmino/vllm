# Power-SMC in vLLM — Implementation Plan

## Summary

Fork vLLM to support Power-SMC's per-token SMC sampling. **Core insight: PagedAttention's `fork()` is almost exactly SMC resampling** — it's a pointer copy, not a data copy. No existing PRs or issues propose SMC in vLLM, making this a novel contribution opportunity.

Reference HF implementation: `/leonardo_scratch/fast/iGen_train/mcimmino/power-smc`

---

## Power-SMC Requirements vs vLLM Capabilities

| Requirement | vLLM Status | Details |
|-------------|-------------|---------|
| Per-token logit access | **Available** | Sampler pipeline exposes raw + processed logprobs. Custom logits processors injectable without forking. |
| Custom sampling (tau=1/alpha) | **Available** | `SamplingParams.temperature` supports arbitrary values per-request. |
| Base-model log-prob alongside sampled token | **Workaround needed** | Custom logits processor can capture raw logits before temperature is applied, compute `log_softmax(logits)`, store in side channel. |
| Mid-generation KV cache reindexing | **Requires modification** | Primitives exist (`fork()`, `free()`) but no hook to call them between decode steps. |
| N=32 parallel particles | **Partially available** | `n=` parameter spawns parallel sequences sharing prompt KV cache. Cannot resample mid-generation. |

---

## vLLM's KV Cache Architecture (Why It's a Good Fit)

The `SelfAttnBlockSpaceManager` provides the exact primitives needed:

| Operation | Method | What it does |
|-----------|--------|-------------|
| Fork | `fork(parent_seq_id, child_seq_id)` | Child shares parent's block table (copy-on-write) |
| Free | `free(seq_id)` | Releases blocks, decrements ref counts |
| Append | `append_slots(seq_id, num_slots)` | Extends a sequence's cache |
| Block table | `get_block_table(seq_id)` | Returns physical block mapping |

**The critical advantage**: `fork()` is a block-pointer copy, not a tensor copy. For a 2048-token sequence with block_size=16, resampling is 128 pointer copies instead of gigabytes of tensor data. Copy-on-write kicks in only when a sharing sequence writes to a block.

This is the same mechanism used for beam search and parallel sampling. **SMC resampling is architecturally identical to beam search expansion** — duplicate winners, discard losers, continue.

---

## What's Missing: The Resampling Hook

The block manager primitives exist but are called by the **scheduler**, not user code. There is no public API to say "at step t, resample these sequences according to these weights." The plugin system (4 extension groups: platform, engine, model, general) does **not** expose this hook point. A fork is required.

---

## Architecture

### Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│  CLIENT (power-smc)                                          │
│  1. Format prompt, send request with smc_* params            │
│  2. Receive N sequences + log_weights + diagnostics          │
│  3. Extract answers, run voting/SNIS → final answer          │
└────────────────┬─────────────────────────────────────────────┘
                 │  OpenAI-compatible API (or offline LLM call)
┌────────────────▼─────────────────────────────────────────────┐
│  vLLM ENGINE                                                  │
│                                                               │
│  ┌─────────────────────────────────────────────┐              │
│  │  SamplingParams (extended)                   │              │
│  │  - smc_alpha, smc_ess_threshold, etc.       │              │
│  └────────────────┬────────────────────────────┘              │
│                   │                                           │
│  ┌────────────────▼────────────────────────────┐              │
│  │  Sampler (modified)                          │              │
│  │  1. Compute log_p = log_softmax(logits)     │              │
│  │  2. Sample from τ=1/α proposal              │              │
│  │  3. Compute incremental_log_w =             │              │
│  │         logsumexp(α * log_p, dim=-1)        │              │
│  │  4. Attach weight to SamplerOutput          │              │
│  └────────────────┬────────────────────────────┘              │
│                   │                                           │
│  ┌────────────────▼────────────────────────────┐              │
│  │  SMCController (new, called by scheduler)    │              │
│  │  1. Accumulate log_weights per SequenceGroup│              │
│  │  2. Compute ESS                             │              │
│  │  3. If ESS < κN: systematic_resample()      │              │
│  │  4. Return ResamplingPlan(forks, frees)     │              │
│  └────────────────┬────────────────────────────┘              │
│                   │                                           │
│  ┌────────────────▼────────────────────────────┐              │
│  │  Scheduler (modified)                        │              │
│  │  Apply ResamplingPlan:                       │              │
│  │  - BlockManager.fork(winner, new_id)        │              │
│  │  - BlockManager.free(loser)                 │              │
│  │  - Update SequenceGroup membership          │              │
│  └─────────────────────────────────────────────┘              │
│                                                               │
│  Keeps running until ALL particles hit EOS or max_tokens.     │
│  Returns: N sequences + log_weights + diagnostics             │
└──────────────────────────────────────────────────────────────┘
```

### Decode Loop (Per Step)

```
For each decode step:
  1. Forward pass for all N particles (already batched)
  2. Sampler: compute log_p, sample at τ=1/α, compute incremental weight
     → weight = logsumexp(α * log_p, dim=-1)  [scalar per particle, computed in-GPU before discarding full logit tensor]
  3. SMCController.accumulate(incremental_log_w)
  4. SMCController.maybe_resample(log_weights, seq_ids)
  5. If resampling triggered:
     a. BlockManager.fork() to duplicate winners' KV caches (pointer copy)
     b. BlockManager.free() to release losers' KV caches
     c. Update SequenceGroup membership & SequenceStatus
     d. Reset log_weights to uniform
  6. Continue until all particles done
```

---

## Key Design Decisions

### 1. Intercept Point: Scheduler Level (not Model Runner)

The resampling hook goes in the **scheduler**, not the model runner. Rationale:
- The scheduler owns sequence metadata and block tables
- `fork()`/`free()` are scheduler-level operations
- The model runner just executes forward passes — it shouldn't know about SMC logic

In V1, the `EngineCoreProc` drives `schedule()` → `execute_model()` → `update()`. The SMC hook goes between `execute_model()` (which produces sampled tokens + weights) and `update()` (which advances sequence state).

### 2. Weight Computation: Inside the Sampler

The HF reference computes weights as:
```python
incremental_log_w = torch.logsumexp(alpha_t * log_p, dim=-1)  # (N,)
```
This requires the **full logit vector** over the vocabulary. In vLLM, the sampler normally discards the full logit tensor and only returns sampled tokens + top-k logprobs.

**Solution**: Compute `logsumexp(α * log_p)` inside the sampler _before_ discarding the full logit tensor. This produces one scalar per particle — negligible memory, no need to return the full vocab distribution. The scalar weight is attached to `SamplerOutput`.

### 3. Finished Particle Handling

**Problem**: In vLLM, a sequence that emits EOS is immediately marked `FINISHED` and evicted from the running batch. In SMC, finished particles must stay in the batch (the HF code forces EOS on done particles by masking logits).

**Solution**: Let particles finish naturally. On the next resample, replace dead slots by forking live winners — conceptually what resampling does anyway. Dead particle slots are "wasted" for at most `1/κ` fraction of steps on average (between resamples), which is acceptable. This stays close to the existing vLLM lifecycle without adding a custom `SMC_ACTIVE` status.

### 4. API Surface & Boundary

Extend `SamplingParams` with SMC-specific fields:

```python
class SamplingParams:
    ...
    # Power-SMC parameters (all optional, None = disabled)
    smc_alpha: float | None = None          # Power exponent α > 1
    smc_ess_threshold: float = 0.5          # Resample when ESS < κ*N
    smc_alpha_ramp_tokens: int = 0          # Linear ramp from 1→α over this many tokens
```

When `smc_alpha` is set and `n > 1`, the engine activates SMC mode for that request. Temperature is automatically set to `1/α` unless explicitly overridden.

**vLLM returns** all N particle sequences + their final `log_weights` + per-particle diagnostics (ESS trace, resample events, `tokens_since_resample`, unique particle count). The request stays active as long as **at least one particle is still generating**.

**Answer selection (majority vote, SNIS draw, etc.) lives entirely in client code** — vLLM's job ends when the last particle finishes or hits `max_tokens`.

### 5. Multi-Request Batching

vLLM's continuous batching means particles from different SMC requests can coexist in the same batch. Resampling must be **scoped to a `SequenceGroup`** — one request's resampling must not touch another request's sequences. This is naturally enforced by keeping the `SMCController` state per-`SequenceGroup`.

### 6. CUDA Graph Compatibility

Batch size stays constant: dead particles get replaced by forks of live ones, so N is always N. CUDA graphs should work. If resampling changes block table pointers mid-step, we may need an eager-mode fallback for that single step. Benchmark to determine impact.

---

## Implementation Phases

### Phase 0: V1 Engine-Only Prototype (bypass API server)

**Goal**: Prove the fork/free/weight cycle works end-to-end.

Directly drive the `EngineCoreProc` with a hardcoded SMC loop:
1. Load model via V1 engine
2. Submit a single request with `n=N` particles
3. After each decode step, read logprobs from `SamplerOutput`
4. Compute weights, ESS, call `BlockManager.fork()`/`free()` manually
5. Validate outputs match HF reference on a few AIME problems

This avoids dealing with the API server, async scheduling, or clean abstractions. Pure proof-of-concept.

### Phase 1: Sampler Extension

Modify the V1 sampler to:
- Detect `smc_alpha` in `SamplingParams`
- Compute `incremental_log_w = logsumexp(α * log_softmax(logits), dim=-1)` before discarding logits
- Attach the scalar weight to `SamplerOutput` (new field)
- Handle optional α-ramping (linearly interpolate α from 1→target over first N tokens)

No scheduler changes yet — just making weights available.

### Phase 2: SMCController + Scheduler Hook

Implement `SMCController`:
- Maintains per-`SequenceGroup` state: `log_weights`, `tokens_since_resample`, `resample_count`
- `accumulate(seq_group, incremental_log_w)` — updates weights
- `maybe_resample(seq_group) → Optional[ResamplingPlan]` — checks ESS, runs systematic resampling
- `ResamplingPlan` = list of `(fork_parent_id, new_child_id)` + list of `free_seq_id`

Modify scheduler to:
- Call `SMCController.maybe_resample()` after each decode step
- Apply `ResamplingPlan` via `BlockManager.fork()`/`BlockManager.free()`
- Update `SequenceGroup` membership (remove losers, add forked winners)
- Handle `SequenceStatus` transitions correctly

### Phase 3: Response Surface + API

vLLM's responsibility ends at returning raw results. No answer aggregation inside vLLM.

- Expose per-particle final `log_weights` in `CompletionOutput` (new field)
- Expose diagnostics: ESS trace, resample step indices, unique final particle count, `tokens_since_resample`
- Wire `smc_*` fields through the OpenAI-compatible API (`extra_body` or dedicated fields)
- The request keeps running as long as **at least one particle is still generating** (not all EOS / max_tokens)

### Phase 4: Optimization + Benchmarking

- Benchmark vs HF reference for correctness (same weights, same ESS trace on deterministic seeds)
- Benchmark throughput: tokens/sec for N=32 particles, compare vLLM-SMC vs HF-SMC
- Profile resampling overhead (fork/free calls per step)
- Test CUDA graph compatibility, fallback to eager if needed
- Multi-request stress test (multiple concurrent SMC requests)

---

## Client: power-smc over vLLM

The client lives in `/leonardo_scratch/fast/iGen_train/mcimmino/power-smc` (or a thin wrapper). It owns everything downstream of raw particle output.

### Client Responsibilities

1. **Prompt formatting** — apply chat template, optional `נקוד` system message
2. **Send request** — call vLLM with `n=N`, `smc_alpha`, `smc_ess_threshold`, etc.
3. **Receive results** — N sequences + `log_weights` + diagnostics per request
4. **Answer extraction** — `extract_aime_answer()` (boxed, fallback to last integer)
5. **Answer selection** — `majority_vote()`, `weighted_majority_vote()`, `snis_draw()`, `snis_draw_active()`
6. **Evaluation** — compare to ground truth, aggregate accuracy

### Example Client Usage

```python
from openai import OpenAI
from power_smc_client import extract_aime_answer, majority_vote, snis_draw

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Send SMC request — vLLM handles all the particle mechanics
response = client.completions.create(
    model="Domyn-Small-v0.2-bf16",
    prompt=prompt,
    n=32,                          # 32 particles
    max_tokens=8192,
    extra_body={
        "smc_alpha": 3.0,         # power exponent
        "smc_ess_threshold": 0.5, # resample when ESS < 0.5*N
        "smc_alpha_ramp_tokens": 100,
    },
)

# Client-side: extract answers + select
sequences = [c.text for c in response.choices]
log_weights = [c.smc_log_weight for c in response.choices]  # new field
answers = [extract_aime_answer(s) for s in sequences]

# Method 1: majority vote (unweighted — particle counts approximate π_α)
answer_mv = majority_vote(answers)

# Method 2: SNIS draw (sample one particle proportional to weights)
answer_snis = snis_draw(answers, log_weights)
```

### Test Prompts (AIME 2025 Style)

For the Phase 0 prototype and correctness validation, use these two problems — one medium combinatorics, one harder number theory — to compare vLLM-SMC output against the HF reference:

**Test Prompt 1 — Combinatorics (expected answer: 3600)**

```
Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\boxed{ANSWER}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

Eight people, including Alice and Bob, are to be seated around a circular table. Two seatings are considered the same if one is a rotation of the other. How many seatings are there such that Alice and Bob are NOT adjacent?
```

Ground truth: **3600**. Standard AIME-style circular permutation with constraint. Tests whether SMC sharpening concentrates particles on correct reasoning chains.

**Test Prompt 2 — Number Theory (expected answer: 10)**

```
Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\boxed{ANSWER}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

Find the number of positive integers n ≤ 1000 such that n can be expressed as the difference of two consecutive perfect cubes.
```

Ground truth: **10** (cubes: 1, 7, 19, 37, 61, 91, 127, 169, 217, 271... up to 1000). Requires recognizing that consecutive cube differences are 3k²+3k+1 and enumerating. Tests multi-step reasoning where SMC can prune wrong approaches early.

**Test Prompt 3 — Algebra (simpler, for sanity checks)**

```
Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\boxed{ANSWER}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

Let f(x) = x^2 + 6x + 5. Find the sum of all values of x such that f(f(x)) = 0.
```

Ground truth: **-12**. Nested quadratic — straightforward enough that even pass@1 should work, useful as a baseline sanity check.

### Validation Protocol

For each test prompt, run both backends with the same seed and compare:

| Check | What to compare |
|-------|----------------|
| Weight trace | `log_weights` at each step should match (within float tolerance) |
| ESS trace | Same resample triggers at same steps |
| Resample ancestors | Same ancestor indices (given same seed) |
| Final sequences | Token-level match after resampling alignment |
| Extracted answers | Same answer distribution across N particles |

If all match, the vLLM fork is a correct drop-in replacement for the HF loop.

---

## Mapping: HF Reference → vLLM Components

| HF Reference (`power-smc/src/`) | vLLM Target | Notes |
|----------------------------------|-------------|-------|
| `power_smc.py` — main decode loop | `EngineCoreProc` / Scheduler | Loop moves from Python to engine internals |
| `power_smc.py` — logit computation (`log_softmax`, `logsumexp`) | Sampler | Computed in-GPU before logits discarded |
| `power_smc.py` — `done` mask + EOS forcing | Scheduler (sequence lifecycle) | Let sequences finish naturally; replace slots on resample |
| `resampling.py` — `systematic_resample()` | `SMCController` | Pure algorithm, portable as-is |
| `resampling.py` — `compute_ess()` | `SMCController` | Pure algorithm, portable as-is |
| `resampling.py` — `reindex_kv_cache()` (index_select) | `BlockManager.fork()` + `BlockManager.free()` | **The big win**: pointer copy replaces tensor copy |
| `power_smc.py` — `snis_draw`, `majority_vote`, etc. | **Client-side** (stays in `power-smc`) | Not part of vLLM — consumes returned sequences + log_weights |
| `answer_extraction.py` | **Client-side** (stays in `power-smc`) | Task-specific, no vLLM coupling |
| `PowerSMCConfig` | `SamplingParams` extension | `smc_alpha`, `smc_ess_threshold`, etc. |

---

## Relevant vLLM Issues/PRs

| Issue/PR | Relevance |
|----------|-----------|
| [PR #10980](https://github.com/vllm-project/vllm/pull/10980) — V1 parallel sampling (sampler) | Foundation for N-particle generation |
| [PR #13421](https://github.com/vllm-project/vllm/pull/13421) — V1 parallel sampling (engine) | Shows how V1 spawns N requests from one prompt |
| [Issue #6226](https://github.com/vllm-project/vllm/issues/6226) — Drop beam search RFC | Discusses whether fork/resampling should remain in V1 |
| [Issue #16802](https://github.com/vllm-project/vllm/issues/16802) — Custom args in completion requests | Could pass SMC params (alpha, kappa, N) via API |
| [Issue #17191](https://github.com/vllm-project/vllm/issues/17191) — Custom sampling params RFC | Prerequisite for custom sampler logic |
| [Issue #25672](https://github.com/vllm-project/vllm/issues/25672) — Generalized KV cache reuse | More flexible cache manipulation patterns |

**No SMC/particle filtering proposals exist in vLLM's issue tracker.**

---

## Key Risks

| Risk | Mitigation |
|------|------------|
| Scheduler complexity — `SequenceGroup` membership changes mid-generation | Phase 0 prototype validates before full integration |
| CUDA graphs break on dynamic block table updates | Eager-mode fallback for resampling steps; benchmark impact |
| Weight computation requires full vocab logits in sampler | Compute `logsumexp` in-place, return scalar — no memory overhead |
| Multi-GPU (device_map) complicates ancestor tensor placement | HF reference already handles this (ancestors moved to layer device) |
| Continuous batching interaction — resample must not cross request boundaries | Scope `SMCController` state per `SequenceGroup` |

---

## V1 Architecture Discovery (Critical for Implementation)

**V1 is fundamentally different from V0** in how it handles parallel sequences. The initial plan assumed V0-style `SequenceGroup` with `fork()`/CoW — this does NOT exist in V1.

### V1 Key Differences

| Concept | V0 | V1 |
|---------|----|----|
| Parallel sampling | `SequenceGroup` with N `Sequence` objects sharing KV via `fork()` | N independent `Request` objects created by `ParentRequest` |
| KV cache sharing | Explicit `fork()` with CoW (block pointer copy) | Implicit via **prefix caching** — same-prefix requests share blocks via `ref_cnt` / `touch()` |
| Block manager | `SelfAttnBlockSpaceManager` with `fork(parent, child)` | `KVCacheManager` → `KVCacheCoordinator` → `BlockPool` (no fork, uses `ref_cnt`) |
| Resampling primitive | `fork()` + `free()` on sequence IDs | Must **abort losing requests + create new requests** that hit prefix cache for winner's tokens |
| Scheduler | Per-`SequenceGroup` scheduling | Per-`Request` scheduling, flat list |

### Implications for Power-SMC

1. **No `fork()` to exploit** — V1's block pool uses `ref_cnt` for sharing, not explicit fork. Two requests sharing the same prefix will naturally share blocks through prefix caching.

2. **Resampling strategy changes**: Instead of "fork winner, free loser" at the block level, we:
   - Abort loser requests
   - Create new requests with winner's full token sequence as prompt
   - Prefix caching ensures the winner's KV blocks are reused (ref_cnt bump, not tensor copy)
   - This is **functionally equivalent** to fork — just at a higher abstraction level

3. **The weight computation still needs to go in the sampler** — this hasn't changed. We need `logsumexp(α * log_p)` computed in-GPU before logits are discarded.

4. **Phase 0 prototype works at the LLM/API level** — no engine internals needed. Token-by-token generate with prefix caching handles the "resampling = new request with same prefix" naturally.

### Revised Phase 0 Approach

The Phase 0 prototype (`smc_prototype.py`) uses the vLLM `LLM` class:
- N particles as separate generate calls per step
- `enable_prefix_caching=True` — shared prompt KV is reused across particles
- After resampling, winner particle text becomes the new prefix for forked particles
- Prefix caching ensures resampled particles' KV blocks are shared (ref_cnt)

This is slower than the deep-integration approach but proves correctness. For Phase 1+, the weight computation moves inside the sampler, and resampling is handled by the scheduler creating/aborting requests.

### Revised Architecture for V1

```
┌──────────────────────────────────────────────────────────────┐
│  CLIENT (power-smc)                                          │
│  Same as before — owns answer extraction + voting            │
└────────────────┬─────────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────────┐
│  vLLM V1 ENGINE                                               │
│                                                               │
│  ┌─────────────────────────────────────────────┐              │
│  │  SamplingParams (extended)                   │              │
│  │  - smc_alpha, smc_ess_threshold, etc.       │              │
│  └────────────────┬────────────────────────────┘              │
│                   │                                           │
│  ┌────────────────▼────────────────────────────┐              │
│  │  Sampler (modified — same as before)         │              │
│  │  Compute logsumexp(α * log_p) → scalar       │              │
│  │  weight per request, attach to output        │              │
│  └────────────────┬────────────────────────────┘              │
│                   │                                           │
│  ┌────────────────▼────────────────────────────┐              │
│  │  SMCController (new, sits in EngineCore)      │              │
│  │  1. Track log_weights per parent request     │              │
│  │  2. Receive weights from ModelRunnerOutput   │              │
│  │  3. Compute ESS, decide resampling           │              │
│  │  4. On resample:                             │              │
│  │     - Abort loser requests                   │              │
│  │     - Create new requests with winner tokens │              │
│  │     - Prefix cache → winner KV blocks reused │              │
│  │  5. Reset weights to uniform                 │              │
│  └─────────────────────────────────────────────┘              │
│                                                               │
│  Prefix caching ensures "fork" is efficient:                  │
│  winner's KV blocks get ref_cnt++, no tensor copy             │
└──────────────────────────────────────────────────────────────┘
```
 
## Updates (2026-03-08)

- **Critical V1 discovery:** vLLM V1 does *not* expose a V0-style `fork()` primitive. Parallel sampling is implemented by spawning independent `Request` objects (via `ParentRequest`) and KV sharing is provided by prefix-caching / `BlockPool` ref-counting rather than an explicit fork/free API.

- **Implication:** Resampling must be implemented by *aborting loser requests and creating new requests seeded with winner prefixes* so the prefix cache reuses winner KV blocks (ref_cnt bump). Functionally equivalent to `fork()` but at a higher abstraction.

- **Phase 0 work (done):**
    - Created a Phase 0 prototype `smc_prototype.py` (token-by-token, slow) to validate resampling-by-restart using `enable_prefix_caching=True`.
    - Created a faster Phase 0 v2 `smc_prototype_v2.py` that issues a single batched generation of `N` particles and computes offline importance weights and voting diagnostics. File: `vllm/smc_prototype_v2.py`.

- **Known limitations of Phase 0:**
    - The token-by-token prototype is slow (one `generate()` per particle per step).
    - The batched Phase 0 v2 performs no mid-generation resampling (importance sampling only) but validates weight computation and voting logic.

- **Next technical steps (Phase 1 → Phase 2):**
    1. Extend `SamplingParams` → thread new SMC fields (`smc_alpha`, `smc_ess_threshold`, `smc_alpha_ramp_tokens`).
    2. Modify `vllm/v1/sample/sampler.py` to compute `incremental_log_w = logsumexp(α * log_softmax(logits), dim=-1)` in-GPU and attach the scalar to `SamplerOutput` (new field).
    3. Add an `SMCController` in `EngineCore`/scheduler to accumulate per-request weights, compute ESS, and on resample abort losers + spawn new requests seeded with winners' token sequences (prefix-caching will reuse KV blocks).
    4. Wire per-particle final `log_weights` into `CompletionOutput` and expose diagnostics to the client.

- **Testing plan:**
    - Run `smc_prototype_v2.py` on Leonardo (GPU allocation) with small `n` (4–8) and the AIME-style test prompts to collect baseline weighted voting and SNIS results.
    - After Phase 1 sampler changes, validate incremental weights compare (within numerical tolerance) to HF reference on deterministic seeds.

- **Files created:**
    - `vllm/smc_prototype.py` (Phase 0, token-by-token)
    - `vllm/smc_prototype_v2.py` (Phase 0 v2, batched importance sampling)

If you'd like, I can now: (A) run the `smc_prototype_v2.py` on the cluster with a short smoke test, or (B) start implementing Phase 1 (sampler changes). Which do you prefer?

