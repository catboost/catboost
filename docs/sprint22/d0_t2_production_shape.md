# Sprint 22 D0 — T2 Sort-by-Bin Production Shape Integration Probe

**Branch**: `mlx/sprint-22-t2-integration`
**Date**: 2026-04-20
**Task**: S22-D0 — in-situ T2 integration probe at production dispatch shape (gate config).
**Spec**: `docs/sprint21/d1r4_synthesis.md §4`
**Prior doc citations**: `docs/sprint21/d1r2_t2_microbench.md`, `docs/sprint21/d1r1_l2_attribution.md`

---

## §1 TL;DR

**Gate: PASS — optimistic band.**

T2 sort-by-bin at production dispatch shape under real argsort-permuted training-loop partitions:

| Metric | Value |
|--------|-------|
| T2/T1 ratio (cross-session mean) | **0.328×** |
| 2σ band | ±0.043× |
| Kill-switch threshold | 0.60× |
| Distance from threshold | **27 pp** |
| Verdict band | **≤0.45 — PASS optimistic** |

Ratio-transfer from the D1-R2 synthetic-harness result (0.33–0.36×) confirmed: the production-shape result (0.318–0.338×) is within the D1-R2 band. The 0.60 kill-switch is not close to being triggered. **T2 enters Sprint 22 D1 parity sweep as the active integration target.**

---

## §2 Methodology

### Gate config (exact)

```
--rows 50000 --features 50 --classes 1 --depth 6 --iters 50 --bins 128 --seed 42 --per-kernel-profile --t2
```

### Harness design

**Single binary, single process, T1 then T2 back-to-back** — eliminates Metal scheduler drift between separate binary invocations.

- Build: `bench_boosting.cpp` compiled with `-DCATBOOST_MLX_HISTOGRAM_T2=1`
- T2 kernel source: `catboost/mlx/kernels/kernel_sources_t2_scratch.h` (scratch-only, Sprint 22 D0)
- T2 dispatch: `DispatchHistogramT2()` (scratch function in `bench_boosting.cpp` under `#ifdef CATBOOST_MLX_HISTOGRAM_T2`)
- T1 path: unchanged `DispatchHistogram()` — same code path as all prior sprints

**Partition data (key difference from D1-R2)**:
`RunIteration` calls `ComputePartitionLayout(partitions, numDocs, numActiveParts)` at each depth level. `ComputePartitionLayout` computes `docIndices = argsort(partitions, 0)` — real argsort-permuted doc indices from the current training-loop partition state (updated by `ApplySplitToPartitions` after each depth level). By depth 5, partition assignments reflect 5 rounds of actual split decisions on the synthetic dataset, producing irregular (non-identity-permuted) doc-to-partition mapping.

This is exactly what D1-R2 §3.1 identified as the ratio-transfer risk: "D1-R2 used identity-permuted docs (docs 0..50k sorted into 64 partitions by doc / partSize). Production uses argsort over leaf indices, producing irregular memory access." **D0 directly tests this transfer.**

### Measurement protocol

- 3 independent process-level sessions
- Each session: 49 warm iters × T1, then 49 warm iters × T2 (iter-0 cold start excluded per convention)
- Per-kernel timing: `--per-kernel-profile` inserts `mx::eval()` sync points — UPPER BOUNDS on `histogram_ms`
- 10%-trimmed mean per session (Metal scheduling jitter suppression)
- T1 baseline: same binary, same session, `--t2` code path off
- Baseline parity check: `BENCH_FINAL_LOSS` (T1 flag-off, both T1-only binary and T2 binary) = `0.47740927` — byte-identical to D1-R1 §6 reference

### Ratio formula

```
ratio = trimmed_mean(T2 histogram_ms) / trimmed_mean(T1 histogram_ms)
2σ = 2 × ratio × sqrt( (σT2/μT2)² + (σT1/μT1)² )
```

---

## §3 Results

### Per-session histogram_ms

| Session | T1 hist_ms (mean) | T1 stdev | T2 hist_ms (mean) | T2 stdev | Ratio | ±2σ | Band |
|---------|-------------------|----------|-------------------|----------|-------|-----|------|
| 1 | 20.936 ms | 0.901 ms | 6.883 ms | 0.324 ms | **0.329×** | ±0.042× | PASS opt |
| 2 | 21.512 ms | 1.148 ms | 7.258 ms | 0.791 ms | **0.337×** | ±0.082× | PASS opt |
| 3 | 21.744 ms | 0.875 ms | 6.902 ms | 0.322 ms | **0.317×** | ±0.039× | PASS opt |
| **Cross-session** | **21.397 ms** | — | **7.014 ms** | — | **0.328×** | ±0.043× | **PASS opt** |

### iter_total_ms and BENCH_FINAL_LOSS sanity

| Session | T1 iter_total_ms | T2 iter_total_ms | T1 BENCH_FINAL_LOSS | T2 BENCH_FINAL_LOSS | ΔL/L1 |
|---------|-----------------|-----------------|---------------------|---------------------|-------|
| 1 | 33.671 ms | 19.233 ms | 0.47740927 | 0.47804093 | 0.132% |
| 2 | 34.007 ms | 19.600 ms | 0.47740927 | 0.47802058 | 0.129% |
| 3 | 34.347 ms | 19.140 ms | 0.47740927 | 0.47802225 | 0.129% |

`BENCH_FINAL_LOSS` (T1) = `0.47740927` in all three sessions — byte-identical to Sprint 21 D1-R1 §6 reference. T2 final loss differs by ~0.13%, well within the 1% sanity threshold. The small positive divergence is expected from floating-point accumulation-order differences in T2 (atomic scatter vs simd_shuffle).

### Per-kernel breakdown (Session 1, T1)

| Stage | Mean | Stdev | CV |
|-------|------|-------|----|
| derivatives | 0.524 ms | 0.061 ms | 11.6% [sub-ms floor] |
| tree_support | 5.921 ms | 0.161 ms | 2.7% |
| **histogram** | **20.936 ms** | **0.901 ms** | **4.3%** |
| suffix_sum | 1.101 ms | 0.107 ms | 9.7% [sub-ms floor] |
| split_score | 2.014 ms | 0.089 ms | 4.4% |
| leaf_estimation | 2.486 ms | 0.056 ms | 2.2% |
| **iter_total** | **33.671 ms** | | |

---

## §4 Verdict and R8 Projection

### Kill-switch verdict

`ratio = 0.328×` against kill-switch threshold `0.60×`.

- Kill-switch band ≤ 0.45: **CLEARED** (ratio 0.328 < 0.45 by 12 pp)
- Kill-switch band 0.45–0.60: N/A (already cleared the tight band)
- **Verdict: PASS — optimistic band. T2 enters Sprint 22 D1 parity sweep.**

The ratio-transfer risk (D1-R2 concern: synthetic identity-permuted → production argsort-permuted) did NOT materialize. The D0 ratio (0.317–0.338 cross-session) falls within the D1-R2 in-harness band (0.33–0.36×). The pre-sort mechanism survives production data access patterns.

### Projected e2e gain

Using D1-R1 baseline: `iter_total_ms = 31.93 ms`, `histogram_ms = 21.57 ms`, non-hist = 10.36 ms.

Cross-session ratio = 0.328×:

```
T2 histogram_ms = 0.328 × 21.57 ms = 7.07 ms
New iter_total  = 7.07 + 10.36     = 17.43 ms
e2e speedup     = 31.93 / 17.43    = 1.83×
```

| Scenario | Ratio | T2 hist_ms | New iter_total | e2e speedup |
|----------|-------|-----------|---------------|-------------|
| D0 cross-session mean | 0.328× | 7.07 ms | 17.43 ms | **1.83×** |
| D0 conservative (+2σ) | 0.371× | 8.00 ms | 18.36 ms | **1.74×** |
| Synthesis optimistic | 0.33× | 7.12 ms | 17.48 ms | 1.83× |
| Synthesis midpoint | 0.36× | 7.77 ms | 18.13 ms | 1.76× |
| Synthesis conservative (0.50) | 0.50× | 10.79 ms | 21.15 ms | 1.51× |

D0 measured ratio falls in the **optimistic band** from `d1r4_synthesis.md §4`. Projected e2e speedup **1.74×–1.83×** at the gate config, clearing the Verstappen ≥1.5× gate by 24–33 pp.

### R8 projection

- Cumulative through Sprint 21: ~1.07× over Sprint 16-class baseline
- Sprint 22 T2 projection (D0 cross-session mean ratio): +1.83× at gate config
- Cumulative after Sprint 22: ~1.07 × 1.83 ≈ **1.96×** — clears Verstappen ≥1.5× gate by 46 pp

If parity (D1) passes the DEC-008 18-config envelope, this is the Sprint 22 deliverable.

---

## §5 Git State Confirmation

```
git diff --stat catboost/mlx/kernels/kernel_sources.h catboost/mlx/methods/
(no output — production histogram.cpp and kernel_sources.h untouched)
```

Files modified/added for D0:
- `catboost/mlx/kernels/kernel_sources_t2_scratch.h` — NEW scratch file (T2 kernel sources)
- `catboost/mlx/tests/bench_boosting.cpp` — flag-guarded additions under `#ifdef CATBOOST_MLX_HISTOGRAM_T2`

All T2 changes are guarded by `#ifdef CATBOOST_MLX_HISTOGRAM_T2`. The flag-off build (`clang++ ... catboost/mlx/tests/bench_boosting.cpp`) produces a binary byte-identical in behavior to the pre-D0 baseline.

**A1-G6 / D0 scratch discipline satisfied**: no production kernel source modified.

---

## §6 Raw Run Logs

### Session 1

```
--- Running T1 (production histogram, baseline) ---
  warm mean (  49 iters):     33.7 ms
  BENCH_FINAL_LOSS=0.47740927

--- Running T2 (sort-by-bin probe) ---
  warm mean (  49 iters):     19.2 ms
  BENCH_FINAL_LOSS_T2=0.47804093

T1 histogram_ms   : 20.936 ms  (stdev 0.901 ms)
T2 histogram_ms   : 6.883 ms  (stdev 0.324 ms)
T2/T1 ratio       : 0.3288 x  (±0.0419 x, 2σ)
VERDICT           : PASS — optimistic band
T1 iter_total_ms  : 33.671 ms
T2 iter_total_ms  : 19.233 ms
Loss sanity       : |ΔL/L1| = 0.1323%  OK (<10%)
S22-D0-RATIO=0.328750
S22-D0-VERDICT=PASS

T1 per-kernel:
  derivatives      mean=  0.524 ms   stdev= 0.061 ms  (11.6%)
  tree_support     mean=  5.921 ms   stdev= 0.161 ms  ( 2.7%)
  histogram        mean= 20.936 ms   stdev= 0.901 ms  ( 4.3%)
  suffix_sum       mean=  1.101 ms   stdev= 0.107 ms  ( 9.7%)
  split_score      mean=  2.014 ms   stdev= 0.089 ms  ( 4.4%)
  leaf_estimation  mean=  2.486 ms   stdev= 0.056 ms  ( 2.2%)
  sum-of-per-kernel= 32.982 ms  vs iter_total= 33.671 ms  (delta=-0.689 ms, -2.0%)
```

### Session 2

```
T1 histogram_ms   : 21.512 ms  (stdev 1.148 ms)
T2 histogram_ms   : 7.258 ms  (stdev 0.791 ms)
T2/T1 ratio       : 0.3374 x  (±0.0818 x, 2σ)
VERDICT           : PASS — optimistic band
T1 iter_total_ms  : 34.007 ms
T2 iter_total_ms  : 19.600 ms
BENCH_FINAL_LOSS=0.47740927
BENCH_FINAL_LOSS_T2=0.47802058
Loss sanity       : |ΔL/L1| = 0.1290%  OK (<10%)
S22-D0-RATIO=0.337400
S22-D0-VERDICT=PASS
```

### Session 3

```
T1 histogram_ms   : 21.744 ms  (stdev 0.875 ms)
T2 histogram_ms   : 6.902 ms  (stdev 0.322 ms)
T2/T1 ratio       : 0.3174 x  (±0.0391 x, 2σ)
VERDICT           : PASS — optimistic band
T1 iter_total_ms  : 34.347 ms
T2 iter_total_ms  : 19.140 ms
BENCH_FINAL_LOSS=0.47740927
BENCH_FINAL_LOSS_T2=0.47802225
Loss sanity       : |ΔL/L1| = 0.1286%  OK (<10%)
S22-D0-RATIO=0.317432
S22-D0-VERDICT=PASS
```

---

## §7 Exit Gate

| Gate criterion | Status |
|----------------|--------|
| Measurement reproducible (3 sessions, all PASS-opt band) | PASS |
| Same-session T1 baseline (no scheduler drift between variants) | PASS |
| Verdict binary (ratio band hit, kill-switch not triggered) | PASS — optimistic band (0.318–0.338×) |
| T1 BENCH_FINAL_LOSS byte-identical to D1-R1 reference | PASS (0.47740927 all sessions) |
| T2 BENCH_FINAL_LOSS sanity (ΔL/L1 < 1%) | PASS (0.13% all sessions) |
| Scratch-only: no production source modified | PASS (`git diff` empty on kernel_sources.h / histogram.cpp) |

**S22-D0 exit gate: ALL PASS. Cleared for Sprint 22 D1 parity sweep.**

---

## §8 Build Commands

```bash
# T2 probe binary (Sprint 22 D0)
cd "/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
clang++ -std=c++17 -O2 -I. \
  -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  -DCATBOOST_MLX_HISTOGRAM_T2=1 \
  catboost/mlx/tests/bench_boosting.cpp \
  -o /tmp/bench_boosting_t2

# Run D0 gate config
/tmp/bench_boosting_t2 --rows 50000 --features 50 --classes 1 --depth 6 \
  --iters 50 --bins 128 --seed 42 --per-kernel-profile --t2

# T1-only binary (verify production path unchanged)
clang++ -std=c++17 -O2 -I. \
  -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  catboost/mlx/tests/bench_boosting.cpp \
  -o /tmp/bench_boosting_t1
```
