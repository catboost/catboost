# Sprint 22 D1 — T2 Sort-by-Bin Parity Sweep

**Branch**: `mlx/sprint-22-t2-integration`
**Date**: 2026-04-20
**Task**: S22-D1 — 18-config parity sweep of T2 sort-by-bin kernel against DEC-008 envelope.
**Prior doc citations**: `docs/sprint22/d0_t2_production_shape.md`, `docs/sprint21/d1r4_synthesis.md §3`, `docs/sprint21/d1r2_t2_microbench.md §3`, `.claude/state/DECISIONS.md DEC-008`

---

## §1 TL;DR

**FAIL-STRUCTURAL — Outcome C. Kahan compensation cannot fix this.**

T2 has a structural bug in the two-kernel dispatch sequence (T2-sort → T2-accum) that causes incorrect histogram accumulation starting from the second boosting iteration. All 18 DEC-008 configs fail. ULP deltas range from 1,327 to 2,583,206 (gate is 4 for RMSE/Logloss, 8 for MultiClass). The failure is non-deterministic run-to-run, confirming it is not a floating-point accumulation order issue that Kahan could address — it is a stale-buffer or execution-ordering bug in the T2-sort → T2-accum inter-kernel handoff.

**T2 drops to RESEARCH. Sprint 22 must pivot.**

---

## §2 Config Matrix (DEC-008 Envelope, Transcribed Exactly)

From `DECISIONS.md DEC-008` (verbatim):

> **DEC-008: Parity tolerance envelope — RMSE/Logloss ulp≤4, MultiClass ulp≤8**
>
> **Sprint**: 17 | **Date**: 2026-04-17
> **Chosen**: (c) Loss-specific, derived from Higham γ_8 bound on the 8-term cross-SIMD fold: **RMSE ulp≤4, Logloss ulp≤4, MultiClass ulp≤8**
> **Scope**: Bounded to `approxDim ∈ {1, 3}`, `N ≤ 50k`, 50 iterations, depth 6.

Config matrix: `{1k, 10k, 50k} × {RMSE, Logloss, MultiClass} × {32, 128} = 18 configs`

| # | N     | Loss       | Bins | Classes | DEC-008 threshold |
|---|------:|:-----------|-----:|:--------|:------------------|
| 1 |  1000 | RMSE       |   32 | 1       | ulp ≤ 4           |
| 2 |  1000 | RMSE       |  128 | 1       | ulp ≤ 4           |
| 3 |  1000 | Logloss    |   32 | 2       | ulp ≤ 4           |
| 4 |  1000 | Logloss    |  128 | 2       | ulp ≤ 4           |
| 5 |  1000 | MultiClass |   32 | 3       | ulp ≤ 8           |
| 6 |  1000 | MultiClass |  128 | 3       | ulp ≤ 8           |
| 7 | 10000 | RMSE       |   32 | 1       | ulp ≤ 4           |
| 8 | 10000 | RMSE       |  128 | 1       | ulp ≤ 4           |
| 9 | 10000 | Logloss    |   32 | 2       | ulp ≤ 4           |
|10 | 10000 | Logloss    |  128 | 2       | ulp ≤ 4           |
|11 | 10000 | MultiClass |   32 | 3       | ulp ≤ 8           |
|12 | 10000 | MultiClass |  128 | 3       | ulp ≤ 8           |
|13 | 50000 | RMSE       |   32 | 1       | ulp ≤ 4           |
|14 | 50000 | RMSE       |  128 | 1       | ulp ≤ 4           |
|15 | 50000 | Logloss    |   32 | 2       | ulp ≤ 4           |
|16 | 50000 | Logloss    |  128 | 2       | ulp ≤ 4           |
|17 | 50000 | MultiClass |   32 | 3       | ulp ≤ 8           |
|18 | 50000 | MultiClass |  128 | 3       | ulp ≤ 8           |

This matches the convention established in `docs/sprint17/parity_results.md`, `docs/sprint18/parity_results.md`, and `docs/sprint20/d1_parity.md`.

**Note on task brief vs DEC-008 text**: the task brief stated "RMSE bit-exact required (ULP = 0)". The authoritative DEC-008 text in DECISIONS.md reads "RMSE ulp≤4". This report uses the authoritative DEC-008 criterion (ulp≤4 for RMSE). All 6 RMSE configs fail by orders of magnitude regardless.

---

## §3 Methodology

### Build

Both binaries freshly compiled from HEAD (`4333c82a7e`) at the start of D1:

```bash
# T2-probe binary (T1 + T2 in same process)
clang++ -std=c++17 -O2 -I. \
  -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  -DCATBOOST_MLX_HISTOGRAM_T2=1 \
  catboost/mlx/tests/bench_boosting.cpp \
  -o /tmp/bench_boosting_t2_d1

# T1-only reference binary
clang++ -std=c++17 -O2 -I. \
  -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  catboost/mlx/tests/bench_boosting.cpp \
  -o /tmp/bench_boosting_t1_d1
```

Both built clean with no warnings.

### Per-config method

For each of the 18 configs:
1. **T1 reference**: run T1-only binary (`bench_boosting_t1_d1`), capture `BENCH_FINAL_LOSS`.
2. **T2 probe**: run T2 binary (`bench_boosting_t2_d1 --t2`), same-session T1+T2, capture `BENCH_FINAL_LOSS` (T1 path in T2 binary) and `BENCH_FINAL_LOSS_T2`.
3. **T1 cross-check**: verify T1 loss from T1-only binary matches T1-path loss from T2 binary (identifies any measurement infrastructure issues).
4. **ULP delta**: FP32 ULP distance between T1 reference and T2 final loss.

### Parameters (all configs)

`--features 50 --depth 6 --iters 50 --lr 0.1 --l2 3.0 --seed 42`

### Execution order

RMSE first (fail-fast discipline: if RMSE fails, Kahan analysis is triggered immediately). Then Logloss, then MultiClass.

### ULP computation

FP32 ULP distance: reinterpret both values as uint32, take abs difference. Same definition used in S17/S18/S20 parity sweeps.

---

## §4 Results Table

| N     | Loss       | Bins | T1 loss      | T2 loss      | Abs delta    | ULP delta | Threshold | Pass/Fail |
|------:|:-----------|-----:|-------------:|-------------:|-------------:|----------:|:---------:|:---------:|
|  1000 | RMSE       |   32 | 0.40689126   | 0.44199428   | 3.510e-02    | 1,177,862 | 4         | **FAIL**  |
|  1000 | RMSE       |  128 | 0.46936080   | 0.47110140   | 1.741e-03    |    58,405 | 4         | **FAIL**  |
|  1000 | Logloss    |   32 | 0.34161490   | 0.41860044   | 7.699e-02    | 2,583,206 | 4         | **FAIL**  |
|  1000 | Logloss    |  128 | 0.61407095   | 0.62744278   | 1.337e-02    |   224,342 | 4         | **FAIL**  |
|  1000 | MultiClass |   32 | 0.61065382   | 0.60799098   | 2.663e-03    |    44,675 | 8         | **FAIL**  |
|  1000 | MultiClass |  128 | 0.99084771   | 0.99501383   | 4.166e-03    |    69,896 | 8         | **FAIL**  |
| 10000 | RMSE       |   32 | 0.44631991   | 0.44638219   | 6.228e-05    |     2,090 | 4         | **FAIL**  |
| 10000 | RMSE       |  128 | 0.48231599   | 0.48304346   | 7.275e-04    |    24,410 | 4         | **FAIL**  |
| 10000 | Logloss    |   32 | 0.30072498   | 0.36359265   | 6.287e-02    | 2,109,489 | 4         | **FAIL**  |
| 10000 | Logloss    |  128 | 0.60412812   | 0.60317880   | 9.493e-04    |    15,927 | 4         | **FAIL**  |
| 10000 | MultiClass |   32 | 0.57359385   | 0.55636895   | 1.722e-02    |   288,986 | 8         | **FAIL**  |
| 10000 | MultiClass |  128 | 0.95665115   | 0.94599903   | 1.065e-02    |   178,713 | 8         | **FAIL**  |
| 50000 | RMSE       |   32 | 0.44676545   | 0.44680500   | 3.955e-05    |     1,327 | 4         | **FAIL**  |
| 50000 | RMSE       |  128 | 0.47740927   | 0.47805217   | 6.429e-04    |    21,572 | 4         | **FAIL**  |
| 50000 | Logloss    |   32 | 0.30282399   | 0.36709177   | 6.427e-02    | 2,156,469 | 4         | **FAIL**  |
| 50000 | Logloss    |  128 | 0.60559267   | 0.60668176   | 1.089e-03    |    18,272 | 4         | **FAIL**  |
| 50000 | MultiClass |   32 | 0.56538904   | 0.57125396   | 5.865e-03    |    98,397 | 8         | **FAIL**  |
| 50000 | MultiClass |  128 | 0.94917130   | 0.94502950   | 4.142e-03    |    69,488 | 8         | **FAIL**  |

**Summary: 0 / 18 configs PASS. 18 / 18 configs FAIL.**

ULP delta range: 1,327 (50k/RMSE/32) to 2,583,206 (1k/Logloss/32). All exceed their DEC-008 threshold by factors of 331× to 645,801×.

T1 cross-check: all 18 configs pass — T1-only binary and T2 binary produce identical T1 losses in every config. The measurement infrastructure is clean; the failure is entirely in T2.

---

## §5 Verdict and Root Cause

### Verdict: FAIL-STRUCTURAL (Outcome C)

T2 fails the DEC-008 parity envelope on all 18 configs. The failure is not due to floating-point accumulation order (which Kahan compensation could address). It is a **structural correctness bug in the T2-sort → T2-accum two-kernel dispatch sequence**, confirmed by the following diagnostic experiments:

### Diagnostic evidence

**Experiment 1: Iteration count isolation**

```
--features 1 --rows 50000 --bins 128 --seed 42 --depth 6
```

| iters | T1 loss    | T2 loss        | Verdict     |
|------:|:-----------|:---------------|:------------|
|     1 | 0.53039330 | 0.53039330     | bit-exact   |
|     2 | 0.49367726 | 142.84576416   | CATASTROPHIC|
|     5 | 0.41061914 | 297.05328369   | CATASTROPHIC|
|    10 | 0.33633292 | 1141.63049316  | CATASTROPHIC|

**iters=1 is bit-exact. iters=2 produces T2 loss ≈ 143 (vs T1 ≈ 0.49).** This establishes that the FIRST histogram dispatch is correct and the BUG ACTIVATES ON THE SECOND CALL.

**Experiment 2: Run-to-run non-determinism**

```
--rows 50000 --features 50 --bins 128 --seed 42 --depth 6 --iters 50
```

| Run | T1 loss    | T2 loss    |
|----:|:-----------|:-----------|
|   1 | 0.47740927 | 0.47804454 |
|   2 | 0.47740927 | 0.47803700 |
|   3 | 0.47740927 | 0.47803560 |
|   4 | 0.47740927 | 0.47803465 |
|   5 | 0.47740927 | 0.47803015 |

T1 is perfectly deterministic. T2 varies on every run. The variance is in the 5th decimal place (~0.47803–0.47804), but it is present on every run and distinct each time. This is the signature of non-deterministic execution — consistent with a race condition that produces slightly different wrong answers each time.

**Experiment 3: Small bin count boundary**

```
--features 50 --rows 50000 --bins N --depth 6 --iters 50
```

| bins | T1 loss    | T2 loss    | Deterministic? |
|-----:|:-----------|:-----------|:---------------|
|    2 | 0.05767765 | 0.05767765 | Yes (bit-exact) |
|    4 | 0.32792997 | 0.32792997 | Yes (bit-exact) |
|    7 | 0.39083135 | varies     | No (run 2: 0.39082098 ≠ T1) |
|    8 | 0.39890137 | varies     | No (run 1: 0.39890140, run 3: 0.39890137) |
|   16 | 0.43137935 | 0.43138158 | No (deviates consistently) |
|  128 | 0.47740927 | varies     | No |

Small bin counts (folds ≤ 4) produce bit-exact results for iters=50. The divergence begins at folds=6 (stochastically) and becomes consistent at larger bin counts. This points to a threshold effect in the sort work that affects the race window.

### Root cause hypothesis

The `DispatchHistogramT2` function dispatches T2-sort and T2-accum as MLX lazy array operations. T2-accum takes `sortOut[0]` (sortedDocs) and `sortOut[1]` (binOffsets) as graph inputs. On the first call:
- `sortedDocs` and `binOffsets` are freshly allocated GPU buffers, zero-initialized.
- T2-sort writes correct data; T2-accum reads it. Result: correct.

On the second call (next boosting iteration):
- MLX's buffer pool may reuse the same GPU memory for `sortedDocs` and/or `binOffsets` (same size, same dtype).
- If the new T2-sort has not yet written to the buffer at the time T2-accum reads it, T2-accum reads the PREVIOUS iteration's sorted indices.
- The previous iteration used a different partition layout (different `numActiveParts`, different `docIndices` argsort). Reading old `sortedDocs` entries with the new iteration's `binOffsets` (or vice versa) produces a mismatch of indices and offsets, causing out-of-range reads into the `stats` array.
- `stats[statIdx * totalNumDocs + garbage_docIdx]` where garbage_docIdx >> totalNumDocs = Metal undefined read = large float values → catastrophic loss inflation.

**Supporting evidence for this hypothesis**:
- iters=1 is always correct (no previous iteration's data)
- Catastrophic failure magnitude (T2 loss = 142 vs T1 = 0.49 at iters=2) consistent with stats reads landing on large gradient values from random positions
- Features=1 config (forces only feature-0 range-scan path, no features 1-3 atomics) is equally affected — rules out the atomic accumulation path as the cause
- Non-determinism scales with sort work per bin count: more bins = more GPU time in T2-sort = larger race window for early T2-accum reads

### Why Kahan does not fix this

Kahan compensation is a sequential summation technique that reduces round-off accumulation. It addresses the case where summing many small values loses precision due to catastrophic cancellation. It cannot address:
1. Out-of-order kernel execution (a pipeline hazard, not a numerical precision problem)
2. Buffer reuse / stale data reads (a memory safety issue)
3. Run-to-run non-determinism from Metal GPU scheduling

Kahan would reduce per-bin ULP from ~64 (D1-R2 measurement) to potentially ~4 ULP. But if the `sortedDocs` buffer is stale, per-bin values will be wrong by orders of magnitude regardless of how precisely each individual sum is computed.

### Why D1-R2 appeared clean

D1-R2 (`docs/sprint21/d1r2_t2_microbench.md`) ran a SINGLE histogram dispatch per measurement run. There was no second call, so the stale-buffer bug did not manifest. The max ULP of 64 observed in D1-R2 was from floating-point accumulation order differences on a CORRECT first-call dispatch.

**D1-R2's parity gate ("max ULP 64 << 1024 bound") was valid for a single dispatch. It was not representative of production multi-iteration parity.** The D1-R2 spec acknowledged this risk: "Sprint 22 D1 parity sweep (18-config DEC-008 envelope) must validate end-to-end ULP, not per-bin." D1 confirms that end-to-end ULP fails massively starting from the second iteration.

---

## §6 Raw Run Logs

### RMSE configs (first 3 representative)

```
=== N=1000 RMSE bins=32 ===
  T1 (reference) : 0.40689126
  T1 (in T2 bin) : 0.40689126
  T2             : 0.44199428
  ULP delta      : 1177862  (threshold: 4)  -> FAIL

=== N=10000 RMSE bins=32 ===
  T1 (reference) : 0.44631991
  T1 (in T2 bin) : 0.44631991
  T2             : 0.44638219
  ULP delta      : 2090  (threshold: 4)  -> FAIL

=== N=50000 RMSE bins=128 ===
  T1 (reference) : 0.47740927
  T1 (in T2 bin) : 0.47740927
  T2             : 0.47805217
  ULP delta      : 21572  (threshold: 4)  -> FAIL
```

### Logloss configs (representative)

```
=== N=1000 Logloss bins=32 ===
  T1 (reference) : 0.34161490
  T1 (in T2 bin) : 0.34161490
  T2             : 0.41860044
  ULP delta      : 2583206  (threshold: 4)  -> FAIL

=== N=50000 Logloss bins=128 ===
  T1 (reference) : 0.60559267
  T1 (in T2 bin) : 0.60559267
  T2             : 0.60668176
  ULP delta      : 18272  (threshold: 4)  -> FAIL
```

### MultiClass configs (representative)

```
=== N=1000 MultiClass bins=32 ===
  T1 (reference) : 0.61065382
  T1 (in T2 bin) : 0.61065382
  T2             : 0.60799098
  ULP delta      : 44675  (threshold: 8)  -> FAIL

=== N=50000 MultiClass bins=128 ===
  T1 (reference) : 0.94917130
  T1 (in T2 bin) : 0.94917130
  T2             : 0.94502950
  ULP delta      : 69488  (threshold: 8)  -> FAIL
```

### Diagnostic run: features=1, iteration isolation

```
features=1, bins=128, N=50000, depth=6, seed=42:
  iters=1:  T1=0.53039330  T2=0.53039330  (EXACT)
  iters=2:  T1=0.49367726  T2=142.84576416  (CATASTROPHIC)
  iters=5:  T1=0.41061914  T2=297.05328369  (CATASTROPHIC)
  iters=20: T1=0.29499832  T2=8104.47949219  (CATASTROPHIC)
```

### 5-run determinism check at gate config

```
50k/RMSE/128b/seed=42/iters=50, T2 values:
  Run 1: 0.47804454
  Run 2: 0.47803700
  Run 3: 0.47803560
  Run 4: 0.47803465
  Run 5: 0.47803015
(T1 is 0.47740927 on all 5 runs)
```

---

## §7 Git State Confirmation

```
git status --short catboost/mlx/kernels/ catboost/mlx/methods/
(no output — production sources unmodified)

git status --short
?? docs/sprint22/scratch/
```

T2 kernel sources remain in scratch-only files:
- `catboost/mlx/kernels/kernel_sources_t2_scratch.h` — T2 kernel (added in D0 commit `4333c82a7e`)
- `catboost/mlx/tests/bench_boosting.cpp` — flag-guarded T2 dispatch (added in D0 commit `4333c82a7e`)

`catboost/mlx/kernels/kernel_sources.h` and `catboost/mlx/methods/histogram.cpp` are unmodified.

**A1-G6 / D2-scratch discipline satisfied: no production kernel source modified.**

---

## §8 R8 Impact

D0 projected e2e speedup of 1.74–1.83× at gate config, contingent on D1 parity pass. D1 FAIL means:

- Sprint 22 cannot ship T2 as-implemented
- R8 projection drops from 1.96× (post-T2) to 1.07× (pre-T2, current) until the bug is fixed
- The Verstappen ≥1.5× gate remains unclearable in its current form without T2 or a replacement lever

If the root-cause bug (stale-buffer / inter-kernel ordering) is fixed, D1-R2's per-bin ULP of 64 would need to be re-evaluated for end-to-end compounding. The fix would require:
1. Explicit `mx::eval()` synchronization between T2-sort and T2-accum in `DispatchHistogramT2`, OR
2. A different buffer management strategy that avoids reuse of stale `sortedDocs`/`binOffsets` data

After the fix, D1 must be re-run (full 18-config sweep).

If Kahan compensation is still needed after the ordering fix (to bring 64 ULP per-bin → ≤4 end-to-end), the Kahan cost estimate is +2–3 days and may compress the D0 ratio from 0.328× toward 0.40×. The D0 kill-switch (ratio > 0.60) would still be well-clear.

---

## §9 Next-Step Recommendation

**T2 drops to RESEARCH. This is non-negotiable per "fix-properly-always" standing order.**

The path forward has two sub-options:

### Option A — Fix the ordering bug and re-run D1 (3–5 days)

Fix: add an explicit `mx::eval({sortOut[0], sortOut[1]})` between the T2-sort dispatch and T2-accum dispatch in `DispatchHistogramT2`. This forces the GPU to complete T2-sort before T2-accum begins, eliminating the stale-buffer race.

Expected outcome after fix:
- iters=1 and iters>1 results should match (the ordering fix eliminates the catastrophic failure)
- Per-bin ULP will revert to D1-R2 levels (~64 ULP), which still fails DEC-008
- A second round of D1 is required to measure end-to-end amplification with correct histograms
- If end-to-end ULP still fails, Kahan compensation (features 1-3 atomic path) may be needed

**Kill-switch**: if after the ordering fix, the 50k/RMSE/128b T2 loss divergence from T1 persists beyond 0.5% (comparable to the D0 sanity threshold), the histogram correctness is still wrong and deeper debugging is needed.

### Option B — Sprint 22 pivots (Sprint 23 entry point for T2 fix)

If Option A is not feasible within Sprint 22, record T2 as a RESEARCH item for Sprint 23 with:
- Bug documented: stale-buffer race in T2-sort → T2-accum MLX lazy graph, second iteration
- Fix known: explicit eval barrier between sort and accum
- D1 re-run required after fix
- Kahan fallback plan from `d1r4_synthesis.md §3` available if needed after ordering fix

Sprint 22 then delivers only the D0 performance measurement (ratio 0.328×, documented in `d0_t2_production_shape.md`), with no perf improvement shipped. R8 stays at 1.07×.

**Recommendation**: If this report is delivered before the Sprint 22 sprint window closes, pursue Option A immediately — the fix is a single `mx::eval()` call, and D1 re-run is a known-cost 1-day sweep. Do not close Sprint 22 without at least attempting the ordering fix.
