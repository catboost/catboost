# Sprint 21 D1-R1 — L2 Production-Shape Re-Attribution

**Branch**: `mlx/sprint-21-hist-tg-reduction`
**Date**: 2026-04-19
**Task**: A1 execution D1-R1 — direct mechanism test of lever L2 (stats pre-permute) at production dispatch shape.
**Dependency**: D1-R3 `--per-kernel-profile` infrastructure (commit `ac378d8de6`).
**Discipline**: A1-G6 — kernel variant is scratch/local only; no production kernel source committed.

---

## 1. TL;DR

**Gate**: ≥ 10% `histogram_ms` reduction under zero-gather upper-bound variant.

**Result**: `histogram_ms` **increased by 2.61%** (−0.56 ms) under the zero-gather variant, well-separated from the +10% threshold in the wrong direction.

**Verdict**: **L2 FALSIFIED at production shape.** L2 does NOT enter the Sprint 22 viable-set.

This is the upper bound — real L2 integration costs a per-iteration O(N) permute kernel on top of this. A non-positive upper bound closes the mechanism entirely: a lever whose zero-cost idealization already fails to improve `histogram_ms` cannot improve it with an added permute cost.

---

## 2. Methodology

### 2.1 Approach

Approach (a) from the delegated plan — local edit of `catboost/mlx/kernels/kernel_sources.h`, measure, restore with `git checkout --`. This is the lowest-risk approach: the production dispatch path (all TG counts, bin counts, partition counts) is exercised identically. Approach (b)/(c) (scaffold replication in `docs/sprint21/scratch/`) would require reimplementing `DispatchHistogram` and adds surface area for mistakes that could bias the comparison.

### 2.2 Exact kernel diff (scratch — not committed)

Inside `kHistOneByteSource`, the `if (valid)` block at lines 193–198:

```diff
  if (valid) {
      const uint sortedPos = partOffset + myDocStart + d;
      const uint docIdx    = docIndices[sortedPos];
      packed = compressedIndex[docIdx * lineSize + featureColumnIdx] | VALID_BIT;
-     stat   = stats[statIdx * totalNumDocs + docIdx];
+     stat   = 1.0f;   // L2-probe: constant replaces stats gather
  }
```

Only the `stat = stats[...]` gather is replaced. The `packed = compressedIndex[...]` load, SIMD shuffle broadcast, bin-owner writes, cross-SIMD fold, and writeback paths are unchanged. This is the most surgical possible isolation of the L2 mechanism — the only thing it measures is the cost of the two stats memory transactions (gradients + hessians, selected by `statIdx`).

### 2.3 Dispatch config

Gate config (unchanged from D0, D1-R3, Sprint 19–20):

```
--rows 50000 --features 50 --classes 1 --depth 6 --iters 50 --bins 128 --seed 42 --per-kernel-profile
```

- Multi-TG production dispatch (depth 6, 64 partitions, ~3 docs/thread at T1 shape — NOT the single-TG shape of S19-01c probe D).
- Warm mean over iters 1–49 (iter 0 cold-start excluded).
- 10%-trimmed mean / stdev per bucket (Metal scheduler jitter suppression, from D1-R3 §7).

### 2.4 Caveat

Per-kernel timings are UPPER BOUNDS on non-sync production cost (`mx::eval()` disables kernel overlap). The **reduction ratio** is more robust than absolute ms — both variants are measured under the same sync regime, so overhead cancels to first order.

---

## 3. Results

### 3.1 Raw `histogram_ms` per run

| Variant | Run 1 | Run 2 | Run 3 | Mean | Between-run σ | Within-run σ (mean) |
|---------|-------|-------|-------|------|---------------|---------------------|
| T1 baseline | 21.604 ms | 21.652 ms | 21.456 ms | **21.571 ms** | 0.102 ms | 1.066 ms |
| L2-constants | 22.080 ms | 22.195 ms | 22.124 ms | **22.133 ms** | 0.058 ms | 1.016 ms |

### 3.2 Reduction

| Quantity | Value |
|---|---|
| Raw delta (L2const − T1) | +0.562 ms |
| **Reduction** (T1 − L2const) / T1 × 100% | **−2.61%** (L2const slower) |
| Propagated 1σ (between-run) | ± 0.54% |
| Propagated 1σ (within-run / √49) | ± 0.97% |
| 2σ band (within-run basis) | [−4.55%, −0.67%] |
| Gate threshold | ≥ +10% |
| Distance from threshold | 12.6 pp |

The reduction is statistically well-separated from zero AND from the +10% gate, in the wrong direction. This is not a close call.

### 3.3 Sanity checks

| Check | T1 | L2-constants | Pass? |
|---|---|---|---|
| BENCH_FINAL_LOSS | 0.47740927 | 0.48174170 | ✓ (expected different — variant produces wrong math by design) |
| Match D1-R3 baseline (31.933 ± 0.208 ms iter_total) | 34.09 ms | 34.76 ms | Both ~8% above D1-R3 baseline — Metal scheduler variability on this run session; the **cross-variant delta** is what's under test, not absolute. |
| `histogram_ms` stdev < 5% | 4.9% (pooled) | 4.6% (pooled) | ✓ |
| Propagated error covers 0? | No | No | ✓ Signal separable from noise. |
| Gate-result margin > 2σ | Yes (12.6 pp vs 2σ ≈ 1.94 pp) | — | ✓ No ambiguity. |

---

## 4. Verdict and interpretation

### 4.1 Primary interpretation

**The AGX hides the stats gather effectively at multi-TG production shape.** Replacing the gather with a constant produced **no measurable improvement** — and a slight regression attributable to minor compiler instruction-scheduling differences once the load was removed (or to Metal scheduler variance alone, since the slowdown falls within within-run 2σ).

This generalizes S19-01c probe D's single-TG finding (global-mem loads = ~0% of kernel cost) to the multi-TG depth-6 dispatch. The concern flagged in `d0_attribution.md §6.4` — that AGX memory-hiding at 195 docs/thread might not transfer to ~3 docs/thread — is answered: it transfers.

### 4.2 Mechanism-direct framing

Per the ultrathink-task-planning discipline (feedback memory `feedback_ultrathink_task_planning.md`):

- **Lever mechanism**: L2 eliminates stats-gather by pre-permuting `gradients[]` and `hessians[]` into partition-contiguous layout.
- **Direct test applied**: zero-gather upper bound via `stat = 1.0f`. No proxy.
- **Gate verdict**: upper bound fails to produce any hist_ms reduction.
- **Therefore**: the mechanism cannot deliver hist_ms savings at production shape. The gate is tested at the mechanism level, not via correlation. No proxy ambiguity to resolve.

### 4.3 What this does NOT rule out

- L2's benefit on **other stages** (e.g., if stats pre-permutation happened to help the derivative-computation stage). Out of scope — D1-R1 targets `histogram_ms` only per the plan.
- L2 at **other dispatch shapes** (e.g., shallower trees, smaller datasets). Out of scope — gate config is production.
- Other memory-system levers (tree-search restructure that changes access patterns structurally). Out of scope — covered by D1-R2 (T2) and Sprint 22 research track.

### 4.4 Sprint 22 implication

**L2 is removed from the Sprint 22 candidate set.** Sprint 22 lever ranking (D1-R4 output) should cite this verdict as the basis. The pre-scoped fallback chain in `d2b_design.md §3` and Sprint 21 README §Background lists L2 as Sprint 22/23 candidate — that status is now downgraded to **falsified**. T2 remains pending D1-R2; tree-search restructure remains the research-level option.

---

## 5. Clean-state confirmation (A1-G6 discipline)

```
$ git status --short
(no output — tree clean)

$ git diff -- catboost/mlx/kernels/kernel_sources.h
(no output — kernel source restored)

$ sed -n '197p' catboost/mlx/kernels/kernel_sources.h
            stat   = stats[statIdx * totalNumDocs + docIdx];
```

Scratch binary `bench_boosting_l2const` was deleted after measurement. No kernel source, no test binary, no artifact other than this doc lands on the Sprint 21 branch. **A1-G6 satisfied.**

---

## 6. Raw run logs (appendix)

### 6.1 T1 baseline runs

```
=== T1 RUN 1 ===
  warm mean (  49 iters):     34.3 ms  |  BENCH_FINAL_LOSS=0.47740927
  derivatives      mean=  0.521 ms   stdev= 0.059 ms  (11.3%)
  tree_support     mean=  5.854 ms   stdev= 0.125 ms  ( 2.1%)
  histogram        mean= 21.604 ms   stdev= 1.068 ms  ( 4.9%)
  suffix_sum       mean=  1.048 ms   stdev= 0.064 ms  ( 6.1%)
  split_score      mean=  1.981 ms   stdev= 0.100 ms  ( 5.1%)
  leaf_estimation  mean=  2.476 ms   stdev= 0.055 ms  ( 2.2%)
  sum-of-per-kernel= 33.484 ms  vs iter_total= 34.232 ms  (delta=-0.748 ms, -2.2%)

=== T1 RUN 2 ===
  warm mean (  49 iters):     34.2 ms  |  BENCH_FINAL_LOSS=0.47740927
  derivatives      mean=  0.502 ms   stdev= 0.051 ms  (10.1%)
  tree_support     mean=  5.785 ms   stdev= 0.142 ms  ( 2.5%)
  histogram        mean= 21.652 ms   stdev= 1.024 ms  ( 4.7%)
  suffix_sum       mean=  1.005 ms   stdev= 0.061 ms  ( 6.1%)
  split_score      mean=  1.973 ms   stdev= 0.114 ms  ( 5.8%)
  leaf_estimation  mean=  2.473 ms   stdev= 0.060 ms  ( 2.4%)
  sum-of-per-kernel= 33.389 ms  vs iter_total= 34.136 ms  (delta=-0.747 ms, -2.2%)

=== T1 RUN 3 ===
  warm mean (  49 iters):     33.9 ms  |  BENCH_FINAL_LOSS=0.47740927
  derivatives      mean=  0.486 ms   stdev= 0.047 ms  ( 9.6%)
  tree_support     mean=  5.718 ms   stdev= 0.109 ms  ( 1.9%)
  histogram        mean= 21.456 ms   stdev= 1.105 ms  ( 5.1%)  [WARN: stdev > 5%]
  suffix_sum       mean=  1.018 ms   stdev= 0.082 ms  ( 8.0%)
  split_score      mean=  1.912 ms   stdev= 0.069 ms  ( 3.6%)
  leaf_estimation  mean=  2.461 ms   stdev= 0.059 ms  ( 2.4%)
  sum-of-per-kernel= 33.052 ms  vs iter_total= 33.834 ms  (delta=-0.782 ms, -2.3%)
```

### 6.2 L2-constants runs (scratch kernel, restored after)

```
=== L2CONST RUN 1 ===
  BENCH_FINAL_LOSS=0.48174170  (expected different — zero-gather is wrong math)
  derivatives      mean=  0.515 ms   stdev= 0.040 ms  ( 7.8%)
  tree_support     mean=  5.845 ms   stdev= 0.187 ms  ( 3.2%)
  histogram        mean= 22.080 ms   stdev= 0.873 ms  ( 4.0%)
  suffix_sum       mean=  1.049 ms   stdev= 0.104 ms  ( 9.9%)
  split_score      mean=  1.997 ms   stdev= 0.102 ms  ( 5.1%)
  leaf_estimation  mean=  2.512 ms   stdev= 0.063 ms  ( 2.5%)
  sum-of-per-kernel= 33.999 ms  vs iter_total= 34.771 ms  (delta=-0.773 ms, -2.2%)

=== L2CONST RUN 2 ===
  BENCH_FINAL_LOSS=0.48174170
  derivatives      mean=  0.528 ms   stdev= 0.049 ms  ( 9.3%)
  tree_support     mean=  5.900 ms   stdev= 0.191 ms  ( 3.2%)
  histogram        mean= 22.195 ms   stdev= 1.155 ms  ( 5.2%)  [WARN: stdev > 5%]
  suffix_sum       mean=  1.096 ms   stdev= 0.134 ms  (12.2%)
  split_score      mean=  2.039 ms   stdev= 0.107 ms  ( 5.3%)  [WARN]
  leaf_estimation  mean=  2.496 ms   stdev= 0.080 ms  ( 3.2%)
  sum-of-per-kernel= 34.255 ms  vs iter_total= 34.884 ms  (delta=-0.630 ms, -1.8%)

=== L2CONST RUN 3 ===
  BENCH_FINAL_LOSS=0.48174170
  derivatives      mean=  0.507 ms   stdev= 0.056 ms  (11.1%)
  tree_support     mean=  5.798 ms   stdev= 0.109 ms  ( 1.9%)
  histogram        mean= 22.124 ms   stdev= 1.019 ms  ( 4.6%)
  suffix_sum       mean=  1.033 ms   stdev= 0.070 ms  ( 6.8%)
  split_score      mean=  1.947 ms   stdev= 0.076 ms  ( 3.9%)
  leaf_estimation  mean=  2.488 ms   stdev= 0.042 ms  ( 1.7%)
  sum-of-per-kernel= 33.896 ms  vs iter_total= 34.612 ms  (delta=-0.716 ms, -2.1%)
```

---

## 7. Sprint 21 exit-gate impact

| Gate | Criterion | Status |
|------|-----------|--------|
| A1-G1 | D0 kill-switch executed with production-shape evidence | PASS (`d0_attribution.md`) |
| A1-G2 | D1-R3 instrumentation produces stable per-dispatch timings | PASS (commit `ac378d8de6`) |
| **A1-G3** | **D1-R1 gives a binary L2 verdict at production shape** | **PASS** (this doc — verdict FALSIFIED) |
| A1-G4 | D1-R2 gives a binary T2 verdict at production shape (sort-inclusive) | PENDING |
| A1-G5 | D1-R4 Sprint 22 plan has mechanism-direct gates | PENDING |
| A1-G6 | No kernel source committed on Sprint 21 branch (all variants = local/scratch) | PASS (§5 above) |

3/6 exit gates passed. D1-R2 (T2 micro-bench) next.
