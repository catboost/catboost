# S31-T2-PORT-GREEDYLOGSUM — Verdict

**Branch**: `mlx/sprint-31-iter1-audit`  
**Date**: 2026-04-24  
**Status**: COMPLETE — G2b FAIL; T3b T1-AUDIT fallback triggered

---

## Scope

Port CPU CatBoost's `GreedyLogSum` border-selection algorithm from
`library/cpp/grid_creator/binarization.cpp` into MLX's `QuantizeFeatures`
(`csv_train.cpp:816–889`), replacing the custom percentile-midpoint
equal-frequency algorithm. Also fix latent P5 (`ScaleL2Reg` missing
at `FindBestSplit` and `FindBestSplitPerPartition` call sites).

---

## Gate Results

| Gate | Criterion | Result | Value |
|------|-----------|--------|-------|
| G2a | Borders byte-match (CPU CatBoost GreedyLogSum vs C++ port) | **PASS with qualification** | 84/100 exact match; 16/100 equal-score tie-breaking |
| G2b | ST+Cosine aggregate drift ≤ 2% at S28 anchor (N=50k, seeds 42/43/44) | **FAIL** | 53.03% (baseline 53.30%) |
| G2c | bench_boosting v5 ULP=0 preserved | **PASS** | 0.48231599 = AN-009 anchor; kernel sources diff = 0 bytes |
| G2d | 18-config L2 SymmetricTree parity non-regression | **PASS** | 18/18 cells in acceptance envelope |

---

## G2a Detail

84/100 border-pairs are byte-exact. The 16 mismatches are exclusively
equal-score tie-breaking differences: when the initial bin has an even
number of unique values, the midpoint split produces two positions
(`lb = N//2`, `ub = N//2 + 1`) with algebraically identical log-sum
gain scores (symmetric). The pip CatBoost v1.2.10 binary resolves these
ties differently from the upstream `binarization.cpp` v1.27.0+ source.
This is not an algorithmic error — both choices are valid GreedyLogSum
solutions. The algorithm is faithfully ported.

---

## G2b Analysis

Aggregate drift: 53.03% (3 seeds mean). S28 pre-T2 baseline: 53.30%.
Reduction: 0.27 percentage points — within measurement noise.

| Seed | MLX RMSE | CPU RMSE | Ratio | Drift |
|------|----------|----------|-------|-------|
| 42 | 0.29659600 | 0.19362645 | 1.5318 | 53.18% |
| 43 | 0.29593400 | 0.19357118 | 1.5288 | 52.88% |
| 44 | 0.29566600 | 0.19320460 | 1.5303 | 53.03% |

The GreedyLogSum port did NOT materially reduce ST+Cosine drift.

**Interpretation**: The T1-PRE qualifier fires. The S26-D0 P10 probe
previously found that border divergence causes only 0.06% ratio gap
at L2+RS=0+N=10k. V6's flat N-scaling (b ≈ 0 across 100× N range)
is consistent with both L2/Cosine mechanisms being N-independent. The
53% drift was not caused by quantization border divergence.

---

## P5 ScaleL2Reg (committed separately)

CPU CatBoost calls `ScaleL2Reg(l2Regularizer, sumAllWeights, allDocCount)`
before passing L2 to split-scoring and leaf Newton-step. MLX was passing
`config.L2RegLambda` raw. Fix: `scaledL2 = L2RegLambda * (sumAllWeights / docCount)`.

**Impact at S28 anchor**: No-op (uniform weights → sumAllWeights = docCount).
**Impact in production**: Load-bearing for any non-uniform sample-weight run.

G2d re-run after P5 commit: 18/18 PASS (confirms no regression).

---

## Commits

| SHA | Description |
|-----|-------------|
| `768ee50abd` | T2 port GreedyLogSum into QuantizeFeatures |
| `627b968983` | T2 G2a probe — borders byte-match infrastructure |
| `bfb20d3241` | T2 P5 fix — ScaleL2Reg at all three split/leaf sites |
| `661ef0bc2c` | T2 gate probes — G2b ST+Cosine drift + G2d L2 parity |

---

## Conclusion

T2's primary deliverable (GreedyLogSum port) is correct and ships clean:
G2a qualified-pass, G2c pass, G2d pass. The ST+Cosine 53% drift is NOT
caused by border divergence (G2b FAIL). The T3b T1-AUDIT fallback is
now the next task: instrumented iter=1 dump (parent stats + top-K=5
candidates + winning tuple per layer, CPU vs MLX side-by-side).

Per the pre-authorized qualifier: STOP here. Do NOT attempt further
speculative fixes. S31-T3b re-opens the iter=1 dump harness path
(`COSINE_RESIDUAL_INSTRUMENT` infrastructure already in csv_train.cpp).

---

## Next Step

**S31-T3b — T1-AUDIT**: Per-layer instrumented iter=1 dump. Compare
CPU vs MLX on the exact histogram, partition stats, split candidate
scores, and winning split tuple at every depth level for iteration 1.
This is the only remaining diagnostic path to locate the structural
divergence identified in DEC-036.
