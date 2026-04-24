# Sprint 31 T1-PRE — Cosine Source-Diff Preflight

**VERDICT: (ii) PRE-SPLIT DIVERGENCE — with qualifier**

**Branch:** `mlx/sprint-31-iter1-audit`
**Date:** 2026-04-24
**Author:** @research-scientist (DEC-036 preflight harness)
**Anchor:** N=50000, GrowPolicy=SymmetricTree, ScoreFunction=Cosine, RandomStrength=0, seed=42

---

## 1. Headline

The Cosine gain formula is **algebraically equivalent** between CPU CatBoost and
the MLX port — both compute, per split candidate and per tree leaf L,

```
num(split) = Σ_L   sumD_L² / (W_L + λ)
den(split) = Σ_L   sumD_L² · W_L / (W_L + λ)²
score(split) = num(split) / sqrt(den(split))          [with leaf L = (leftChild, rightChild) pair]
```

summed over approx-dims identically on both sides. No missing regularization
term, no asymmetric parent-subtraction, no K-dim order mismatch, no sign error
that survives the quadratic.

**However, a pre-split input DOES diverge by construction:** MLX's feature
quantization uses a **custom percentile-midpoint equal-frequency algorithm**
(`csv_train.cpp:832–872`), while CPU CatBoost defaults to **`GreedyLogSum`**
(`binarization_options.h:16`; `oblivious_tree_options.cpp:22` confirms
`ScoreFunction=Cosine` is the CPU default). The two algorithms place borders
at different values on real-valued features. Sprint 26 D0 empirically
documented this divergence (P10 probe) at L2+RS=0+N=10k and observed 0.06%
ratio gap with "functionally equivalent" splits.

This fires **kill-switch K4 (pre-split divergence)** in DEC-036. **Qualifier:**
the prior S26-D0 evidence makes it unlikely that border divergence alone
explains the 53.30% ST+Cosine drift at N=50k — see §6 — so K4's
"trivial-class fix expected" clause is doubtful. S31 re-scoping should
treat the border port as a **necessary cleanup** (removes a known structural
confound) while keeping an eye on whether drift persists after the port. If it
persists, T1-AUDIT (iter=1 instrumented dump) is still warranted.

A secondary finding: at the CatBoost default `l2_leaf_reg = 3.0` and
`sumAllWeights/docCount = 1.0` (no sample weights in S28 anchor), the CPU
`scaledL2Regularizer` simplifies to `L2RegLambda` identically; MLX's failure
to call `ScaleL2Reg` is a latent divergence that **does not fire** for the S28
anchor but would fire the moment sample-weights or per-class weighting are
introduced.

---

## 2. Formula mapping table — CPU vs MLX (Cosine gain at the SymmetricTree call site)

| # | CPU term (authoritative) | CPU file:line | MLX term | MLX file:line | Equivalent? | Notes |
|---|---|---|---|---|---|---|
| F1 | per-leaf avg: `avg_L = sumD_L / (W_L + scaledL2)` | `online_predictor.h:112-119` (`CalcAverage`) | `dInvL = 1/(dWL + dL2)` → implicit `sumLeft · dInvL` | `csv_train.cpp:1330,1502,1728,1812` | yes (algebraic, fp64 on both) | CPU's `count>0` guard = MLX's `weightLeft<1e-15f continue` — guard semantics differ (strict > 0 vs ≥ 1e-15) but neither admits the numerical case the other rejects at RMSE-unweighted |
| F2 | numerator term added: `avg_L · sumD_L` ≡ `sumD_L² / (W_L+λ)` | `short_vector_ops.h:77, 79` (`UpdateScoreBinKernelPlain`) | `dSL · dSL · dInvL` (+R twin) | `csv_train.cpp:1332, 1504, 1730, 1814` | yes | Expanded form eliminates the divide in the numerator, but is algebraically identical to CPU's `avg · sumD` form |
| F3 | denominator term added: `avg_L² · W_L` ≡ `sumD_L² · W_L / (W_L+λ)²` | `short_vector_ops.h:78, 80` | `dSL · dSL · dWL · dInvL · dInvL` (+R twin) | `csv_train.cpp:1333-1334, 1505-1506, 1731-1732, 1815-1816` | yes | Same expansion as F2 — the two terms are the second moment of `avg_L · sqrt(W_L)` about zero |
| F4 | L2 added to denominator in leaf-avg only — **no additive L2 on Σ, no L2 penalty on score** | `short_vector_ops.h:68, 77` | Same: `λ` appears only in `(W+λ)` and `(W+λ)²` | `csv_train.cpp:1319-1336, 1487-1520, 1715-1738, 1799-1820` | yes | Cosine has no separate L2-penalty term on the aggregate score — only the leaf-value ridge. Confirmed symmetric. |
| F5 | parent-gain subtraction | not present in `TCosineScoreCalcer` | not present in MLX Cosine branch | — | yes (both absent) | L2 branch has `- totalSum²/(totalWeight+λ)` on both sides; Cosine branch has no such term on either side. Documented in `csv_train.cpp:1185` |
| F6 | summation over leaves: `Σ_L (num_L, den_L)` into a single per-split pair, finalize with `num/sqrt(den)` **once** | `score_calcers.h:58, 64-66`; `scoring.cpp:673-683` (leaf loop calls `UpdateSplitScore` → `AddLeafPlain`) | `for (p in partitions) { for (k in K) { cosNum_d += ...; cosDen_d += ...; } } ; gain = cosNum_d / sqrt(cosDen_d)` | `csv_train.cpp:1304-1348, 1453-1530` | yes | Single-division-at-end matches. Accumulation order is `(partition-major, dim-minor)` on MLX; CPU calls `UpdateScores` once per dim per leaf — the final sum is additive so order is algebraically irrelevant (ignoring fp64 rounding, which is the same on both sides given both now use fp64 accumulators) |
| F7 | K-dim aggregation: all dims feed the same `Scores[splitIdx]` pair | `scoring.cpp:751-766` — `for dim in approxDimension` calls the same `scoreCalcer` which accumulates additively | `for (k in K) { cosNum_d += ...; cosDen_d += ...; }` | `csv_train.cpp:1305-1340, 1475-1523` | yes | Both flatten K-dim into a single (num, den) accumulator before the sqrt |
| F8 | denominator initial guard | `1e-100` (fp64) in `Scores.resize` | `1e-20` (fp64) in `cosDen_d` | `score_calcers.h:52` vs `csv_train.cpp:1289, 1430, 1704, 1789` | yes (practically) | At N=50k gradient Σ is O(10³), so `den` ≈ O(10⁴) — both guards are ≥10 orders of magnitude below, neither affects argmax |
| F9 | gain sign convention at argmax | `score = num/sqrt(den)`; **maximized** via `-score` cost in `CalcScoresForBestCandidate` | `Score = -totalGain`, **minimized** (`Score` is a cost); `Gain = totalGain` used for internal argmax | `greedy_tensor_search.cpp` (CPU); `csv_train.cpp:1361-1362, 1571-1572, 1754-1756` | yes | Both treat higher score = better; MLX stores negated `Score` field only for downstream compatibility with the CatBoost convention; the internal argmax uses `bestGain > previous` i.e. maximize. No sign error. |
| F10 | stats container signs | `SumWeightedDelta = Σ weightedDer[doc]`, where `weightedDer = CalcDer · sampleWeight`; CPU `CalcDer(RMSE) = target − approx` (positive residual) | `dimGrads[0] = approx − target` (negated residual) | CPU `error_functions.h:393`; MLX `csv_train.cpp:3462` | **yes for Cosine** (quadratic kills the sign) | Uniform sign flip on `sumD` cancels in `sumD²/(W+λ)` numerator AND `sumD²·W/(W+λ)²` denominator. Cosine score is invariant under global gradient sign flip. Would matter for L2 leaf-value estimation — but that is a separate audit. |
| F11 | stats container: weights | `SumWeight = Σ sampleWeights[doc]` — the *sample weight*, not the hessian | `weightRight, weightLeft` = Σ `dimHess[k][doc]` per bin — the *hessian* | CPU `scoring.cpp:217`; MLX `csv_train.cpp:3780, 3967` (via histogram with `dimHess[k]` in stat-slot-1) | **yes for unweighted RMSE only** | For RMSE hess = 1 always (`csv_train.cpp:3463`), sampleWeights = 1 (S28 anchor has no sample weights), so both reduce to `docCount`. Latent divergence for loss functions with non-trivial hess (Logloss σ(1-σ), Poisson exp, Tweedie, Multiclass) — MLX plugs hessian where CPU plugs weight. For those losses, Cosine scores would numerically diverge. Out of scope for S28 anchor. |

**Verdict on formula:** All 11 rows "yes" for the S28 anchor cell. **No formula-level divergence.** This eliminates verdict (i).

---

## 3. Pre-split preflight table

| Row | Input | CPU path | MLX path | Equivalent? | Evidence |
|-----|-------|----------|----------|-------------|----------|
| P1 | basePred (initial constant approx) | `CalcOneDimensionalOptimumConstApprox(RMSE)` → `CalculateWeightedTargetAverage` = Σw·y / Σw (fp64) | `CalcBasePrediction(rmse)` = Σw·y / Σw (fp64), cast to float32 on store | **yes** (algebraically; MLX has a single fp64→fp32 cast at write time) | `optimal_const_for_loss.h:188-189, 17-29` vs `csv_train.cpp:3142-3150` |
| P2 | Initial per-doc gradient at iter=1 for RMSE | `CalcDer = target − approx` → `weightedDer = der · sampleWeight` → `SumWeightedDelta` accumulator | `dimGrads[0] = approx − target` (**negated sign**) → histogram-accumulated | **sign-flipped**, quadratic-invariant for Cosine | `error_functions.h:393` vs `csv_train.cpp:3462` |
| P3 | Initial per-doc hessian at iter=1 for RMSE | `CalcDer2 = -1` (stored in `SumDer2` — not used for Cosine score) | `dimHess[0] = +1` (sign-flipped and used as "weight" for score) | divergent value + divergent role — **but cancels for unweighted RMSE** | `error_functions.h:381, 396-398` vs `csv_train.cpp:3463` |
| P4 | `SumWeight` semantics at score-calc time | `Σ sampleWeight[doc]` (= 1 when no `sample_weight`) | `Σ dimHess[k][doc]` (= 1 for RMSE) | **same numeric value**, different algebraic object; equivalent for RMSE only | `scoring.cpp:217` vs `csv_train.cpp:3780, 3967` |
| P5 | L2 regularizer value seen by score calcer | `scaledL2Regularizer = L2RegLambda · sumAllWeights / docCount` | raw `config.L2RegLambda` passed directly (no scaling) | **latent divergence**; equal at the S28 anchor because `sumAllWeights/docCount = 1.0` (no per-doc weights) | `scoring.cpp:749, 1031` vs `csv_train.cpp:4068, 4189` |
| P6 | **Feature quantization borders (default EBorderSelectionType)** | **`GreedyLogSum`** (CatBoost default) via `NCB::BestSplit` in `library/cpp/grid_creator/binarization.cpp` | **Custom percentile-midpoint** `frac = (b+1)/(numBorders+1); idx = frac·(nUnique-1); border = 0.5·(sorted[idx]+sorted[idx+1])` | **NO — divergent by construction** | `binarization_options.h:16`, `data_processing_options.cpp:15` (CPU); `csv_train.cpp:851-865` (MLX). Sprint 26 D0 probe P10 confirmed empirically. |
| P7 | Feature quantization bin assignment | `upper_bound(borders, value)` (CPU) | `upper_bound(borders, value)` (MLX) — same logic | **yes**, on identical borders | identical on both |

**Verdict on pre-split:** P1, P4, P7 equivalent. P2, P3 are sign flips that the Cosine quadratic absorbs. P5 is latent but does not fire at the anchor. **P6 is a hard structural divergence that does fire at the anchor.**

This maps to DEC-036's row *"Any layer with same (feature, bin) but different gain value"* — except that here the (feature, bin) grids themselves don't align between CPU and MLX.

---

## 4. Sign-flip reconciliation (F10/P2/P3)

CPU RMSE: `CalcDer = target − approx`, `CalcDer2 = −1`. MLX RMSE:
`dimGrads = approx − target` (sign flipped w.r.t. CPU), `dimHess = +1` (sign
flipped w.r.t. CPU but then used as "weight" in the score, where CPU uses
`sampleWeight = +1`). Algebraic confirmation that Cosine score is invariant:

Let `s_CPU = +Σ der[doc]` and `s_MLX = −Σ der[doc]`, `W = +1 · N_leaf` on both
sides. Then

```
num_CPU = s_CPU²/(W+λ) = s_MLX²/(W+λ) = num_MLX      (square kills sign)
den_CPU = s_CPU²·W/(W+λ)² = s_MLX²·W/(W+λ)² = den_MLX
```

so `num/√den` is bit-identical modulo fp64 rounding of the common subexpression.
This sign flip **does** matter for L2 leaf-value estimation (the leaf value
would differ by sign, breaking trajectory) — but that path goes through
different code (`approx_calcer.cpp`, not `score_calcers.h`) and is out of
scope for split-selection preflight.

---

## 5. Mechanism naming (P6 — the border divergence)

**Mechanism**: MLX's feature quantization algorithm is a hand-written
percentile-based equal-frequency binarizer that is not the same algorithm as
CatBoost's `GreedyLogSum` default. At N=50k with 128 bins, a real-valued
feature's borders will differ between the two implementations. Since the
histogram accumulator bins docs into those borders, the `(feature, bin) → (sumL,
wL, sumR, wR)` tuples that the Cosine score consumes are **not computed over
the same partitioning of the feature axis**. Two different candidate-split
populations are being argmax'd.

**Mapping to DEC-036 classes:** this is a **pre-split input divergence**
(feature quantization borders), not a formula-level divergence — the bucket
**values** on both sides are computed with identical formulas, but the buckets
themselves span different real-value intervals. Fires **K4**.

**File:line pointers:**
- MLX quantizer: `catboost/mlx/tests/csv_train.cpp:816–889` (`QuantizeFeatures`)
- MLX self-documents the divergence at `catboost/mlx/tests/csv_train.cpp:3410`
  (`[DBG iter=0] FeatureBorderType = EqualFrequency (custom MLX impl)`)
- CPU default: `catboost/private/libs/options/binarization_options.h:16`
  (`borderSelectionType = EBorderSelectionType::GreedyLogSum`)
- CPU selector dispatch: `library/cpp/grid_creator/binarization.cpp:114-131`
  (`MakeBinarizer`)
- Prior empirical evidence: `docs/sprint26/d0/split-selection-root-cause.md`
  §2 (P10 probe) — 0.06% ratio gap at L2+RS=0+N=10k, falsified as *dominant*
  cause

---

## 6. Qualifier — why K4 may not close S31

The K4 kill-switch in DEC-036 states: *"re-scope S31 to a pre-split fix
track; T1-AUDIT deferred. Trivial-class fix expected."* This qualifier
questions "trivial-class":

1. **Prior empirical:** S26-D0 ran CPU-vs-MLX at L2+RS=0+N=10k and found the
   border divergence produces only a 0.06% ratio gap, with CPU's chosen split
   at MLX rank 2 (gain delta 0.013% within MLX's own candidate ordering). Not
   dominant for L2.

2. **V6 N-scaling evidence:** drift is *N-independent* (b ≈ 0.0017 across a
   100× range). A border-mismatch mechanism should see drift **decrease with
   N** as border spacing tightens and mismatches span smaller feature
   intervals. Flat scaling does not cleanly predict border-driven drift.

3. **Hypothesis:** porting `GreedyLogSum` closes a known structural confound
   (necessary cleanup) but may not move the 53% ST+Cosine trajectory drift.
   If drift persists post-port, T1-AUDIT (the full instrumented iter=1 dump)
   remains the next investigation.

4. **Counter-hypothesis worth holding:** Cosine scores are much more sensitive
   than L2 to whether a split separates residuals *cleanly* along a feature,
   because the denominator `Σ sumD²·W/(W+λ)²` penalizes split imbalance
   asymmetrically. It is *possible* that a 0.06% L2-ratio border mismatch
   maps to a much larger Cosine-ratio mismatch at 50k — but this is a
   guess, not something this preflight measured.

**Recommended posture:** file the K4 fire and scope a border-port as S31-T2,
but keep T1-AUDIT warm as S31-T3 contingent on the port not closing the drift.

---

## 7. Secondary finding — latent L2 scaling divergence (P5)

At the S28 anchor (`sample_weight = None`, RMSE), CPU
`scaledL2Regularizer = L2RegLambda · 1.0 = L2RegLambda`, equal to MLX's raw
value. **However:** the moment users set per-sample weights or use a loss
with non-uniform hessian in CPU (e.g., Logloss where `sumAllWeights` is
computed over actual sample weights), the two paths diverge.

MLX currently passes `config.L2RegLambda` unchanged (`csv_train.cpp:4068,
4189`); CPU always scales by `sumAllWeights/docCount`. This is a correctness
bug waiting to fire whenever someone uses `sample_weight`. It does not fire at
the S28 anchor and therefore does not alter this verdict, but should be
tracked (e.g., new ticket under the DEC-036 umbrella or a separate
correctness issue).

---

## 8. Next-step recommendation

**Fire K4 and re-scope S31** to port CPU `GreedyLogSum` (and an
`EqualFrequency` fallback matching CPU's spec, not the current custom MLX
variant) into `QuantizeFeatures` in `csv_train.cpp`. Gate the port with a
parity re-run at the S28 anchor: if ST+Cosine drift closes from 53.30% to
≤1% at N=50k, done. If drift persists, escalate to T1-AUDIT per DEC-036's
original plan. Also track the latent `ScaleL2Reg` omission (P5) as a
follow-up correctness item.

---

## 9. Summary

- **Formula**: algebraically equivalent. No divergence at any of 11 audited rows.
- **basePred / gradient / hessian / L2**: equivalent-for-anchor (sign flips absorbed by quadratic; L2 scaling identity at unweighted anchor).
- **Quantization borders**: **divergent by construction**. MLX's custom percentile-midpoint algorithm is not CatBoost's `GreedyLogSum` default.
- **Verdict**: **(ii) PRE-SPLIT DIVERGENCE**, K4 fires.
- **Qualifier**: K4's "trivial-class fix" assumption is doubted; border divergence was already shown non-dominant at L2+RS=0+N=10k (S26-D0), and V6's flat N-scaling does not predict a border mechanism cleanly. The port is **necessary cleanup** but may not close the drift. Keep T1-AUDIT warm as S31-T3.
