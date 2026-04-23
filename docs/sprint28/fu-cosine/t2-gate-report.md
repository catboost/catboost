# S28-COSINE Gate Report

**Branch**: `mlx/sprint-28-score-function-fidelity`
**Date**: 2026-04-23
**Authored by**: @ml-engineer (S28-COSINE, task #71)
**Raw data**: `docs/sprint28/fu-cosine/t2-gate-results.json`

---

## Context

S28-COSINE (Option A) adds `ComputeCosineGain` and `ComputeCosineGainKDim` as
parallel helper functions in `FindBestSplitPerPartition` (csv_train.cpp:~1272–1345).
The main dispatch path continues to use L2 — Cosine dispatch is S28-L2-EXPLICIT
(task #72). This gate therefore validates the **formula** (G2a-formula) and
**L2 non-regression** (G2b), not yet end-to-end Cosine RMSE parity.

---

## Gate G2a-formula: Cosine kernel formula parity

**Definition**: `ComputeCosineGain(sumL, wL, sumR, wR, λ)` in float32 matches
CPU's `TCosineScoreCalcer::UpdateScoreBinKernelPlain` reference (double precision)
within ≤4 ULP at FP32 resolution (DEC-008 RMSE bound).

**Method**: 10 test cases spanning the typical DW N=1000 regime (balanced splits,
tiny leaves, zero gradients, micro-leaf high-gradient, negative gradients). Each
case compared at `λ=3.0`.

### Per-case results

| Case | MLX_f32 | CPU_f64 | delta_rel | ULPs |
|---|---|---|---|---|
| typical balanced | 0.63350832 | 0.63350837 | 6.7e-08 | 1 |
| unbalanced left-heavy | 0.93764156 | 0.93764158 | 2.1e-08 | 0 |
| tiny left leaf (1 doc) | 0.66819167 | 0.66819164 | 4.3e-08 | 0 |
| symmetric opposite signs | 0.69282031 | 0.69282032 | 1.8e-08 | 0 |
| large symmetric | 1.82574189 | 1.82574186 | 1.6e-08 | 0 |
| near-zero gradients | 0.00200000 | 0.00200000 | 4.8e-08 | 0 |
| micro-leaf high gradient | 9.99984741 | 9.99984776 | 3.5e-08 | 0 |
| zero left gradient | 0.56694674 | 0.56694671 | 6.2e-08 | 1 |
| negative then positive | 1.25338328 | 1.25338330 | 1.7e-08 | 0 |
| unbalanced hessian | 0.89905882 | 0.89905889 | 7.5e-08 | 1 |

**Max ULP: 1** (three cases at 1 ULP; all others 0 ULP)

**Tolerance measurement**: 1 ULP. The CPU implementation uses `double` for all
accumulation; the MLX implementation uses `float32`. The 1-ULP drift is a
standard float32-vs-double rounding artifact on the final `sqrt` division step.
It is within the DEC-008 RMSE/Logloss bound (≤4 ULP) by a factor of 4.

**G2a-formula verdict: PASS** (max 1 ULP, threshold 4 ULP)

---

## G2a end-to-end baseline (informational, not pass/fail at this commit)

Since ComputeCosineGain is added but not yet dispatched (S28-L2-EXPLICIT scope),
the end-to-end RMSE gap between MLX (L2) and CPU (Cosine) is recorded as a
baseline to be closed by S28-L2-EXPLICIT:

| seed | MLX_RMSE | CPU_Cosine | ratio |
|------|----------|------------|-------|
| 42 | 0.181591 | 0.210677 | 0.8619 |
| 43 | 0.180416 | 0.208968 | 0.8634 |
| 44 | 0.182786 | 0.210156 | 0.8698 |
| 45 | 0.182511 | 0.213174 | 0.8562 |
| 46 | 0.182111 | 0.219571 | 0.8294 |

Ratio range: [0.83, 0.87] — the 14-17% gap documented in DEC-032. This baseline
confirms the gap is still present (as expected at this commit) and gives S28-L2-EXPLICIT
a quantitative target: all 5 seeds must reach ratio ∈ [0.98, 1.02] after dispatch is wired.

---

## Gate G2b: L2 regression

**Definition**: MLX DW N=1000 vs CPU DW with `score_function='L2'`, 5 seeds {42–46},
rs=0. Ratios must remain ∈ [0.98, 1.02] — identical to G3-FU3 (S27).

### Per-seed results

| seed | MLX_RMSE | CPU_L2 | ratio | verdict |
|------|----------|--------|-------|---------|
| 42 | 0.181591 | 0.181673 | 0.9995 | PASS |
| 43 | 0.180416 | 0.180156 | 1.0014 | PASS |
| 44 | 0.182786 | 0.182824 | 0.9998 | PASS |
| 45 | 0.182511 | 0.182678 | 0.9991 | PASS |
| 46 | 0.182111 | 0.182548 | 0.9976 | PASS |

**G2b verdict: PASS** (5/5 seeds, max deviation 0.24% at seed=46)

No L2 regression from adding `ComputeCosineGain`. The new helper function is
not invoked from any execution path — it cannot affect L2 output by construction.
The build was recompiled from source to confirm no stale object contamination.

---

## REFLECT

### Algorithmic surprises

1. **Cosine picks measurably different splits at 20% rate.** Out of 200 randomly
   sampled 15-doc partitions (synthetic, normal gradients), Cosine and L2 disagree
   on the best split in 40/200 cases (20%). This is higher than anticipated. The
   implication: S28-FU3-REVALIDATE should expect measurably different tree structures
   when Cosine is dispatched — not just RMSE shifts, but different feature rankings
   and split thresholds. Ramos should expect the FU-3 revalidation to show structural
   model differences, not just numeric drift.

2. **L2 regression is clean.** No shared state contamination. The Option A design
   (parallel helper, not invoked) is safe with no surprises.

### Float32 precision observations

Max ULP = 1. The formula is well-conditioned for the DW N=1000 regime:
- Denominator guard `1e-20f` vs CPU's `1e-100` is negligible (tested numerically in
  the near-zero gradients case: ULP=0).
- `sqrt` precision: the 1-ULP cases arise from the final `num / sqrt(den)` step
  where float32 rounding of `sqrt` differs from double by at most 0.5 ULP
  internally, then rounded again to float32. All within FP32 spec.
- No Kahan-style compensated accumulation needed for K≤3, N≤15.

### Known risks for S28-L2-EXPLICIT (task #72)

1. `FindBestSplitPerPartition` accumulates gains across `K` approx dimensions in a
   single `gain` variable before comparing. The Cosine gain accumulation must also
   accumulate `num` and `den` separately across K before computing `num/sqrt(den)`.
   `ComputeCosineGainKDim` is provided for this purpose. S28-L2-EXPLICIT must ensure
   it calls the K-dim accumulation correctly, not the per-leaf scalar overload.

2. The Cosine score is an absolute metric (no parent subtraction), while L2 is
   differential. Ranking is monotone (both higher-is-better), but the lack of
   baseline subtraction means Cosine scores are not zero-centered. This is fine for
   `bestGains[p]` comparison but must not be confused with "gain" in the model JSON
   output — the `Gain` field should remain L2-style after S28-L2-EXPLICIT.

---

## Overall gate verdict

| Gate | Criterion | Result |
|------|-----------|--------|
| G2a-formula | Max 1 ULP (float32 vs CPU double) | **PASS** |
| G2b-L2-regress | 5/5 seeds ∈ [0.98, 1.02] | **PASS** |
