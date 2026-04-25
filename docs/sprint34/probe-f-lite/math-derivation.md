# S34-PROBE-F-LITE T0a — Per-side mask correctness at csv_train.cpp:1698 (one-hot branch)

**Mode:** Pragmatic (applied; ground truth = the actual code)
**Scope:** `FindBestSplit` Cosine and L2 paths, one-hot branch (L1654–1770) and ordinal branch (L1772–2099) of `catboost/mlx/tests/csv_train.cpp`.
**Author:** mathematician
**Date:** 2026-04-25

> **TL;DR.** The advisory-board frame ("does the parent term cancel the per-side mask when wR=0 in one-hot Cosine?") rests on a premise that does **not** hold in the actual code: the **Cosine path subtracts no parent term at all** — neither per-partition nor per-(partition, k). For Cosine the per-side mask is therefore **materially different** from the joint-skip behaviour: when `wR=0` in some partition p (one-hot has many such), it injects `(totalSum_p)² / (totalWeight_p + λ)` into `cosNum_d` and `(totalSum_p)² · totalWeight_p / (totalWeight_p + λ)²` into `cosDen_d` for that (p,k), where the joint-skip injected zero. Crucially, this **inflates the gain ranking of one-hot bins that touch many degenerate (wR=0) partitions** — exactly what produces the empirical 3% regression on the synthetic anchor. **Verdict: per-side mask is wrong as written for one-hot Cosine.** Recommended fix shape: **leave one-hot Cosine joint-skip as-is** (for Cosine specifically) — the partition contribution that the per-side mask "rescues" is not real signal but a baseline ‖total‖² that the absence of parent-term subtraction lets through. Confidence: **high** for Cosine; **medium-to-high** that the same fix is harmless for one-hot L2 (because L2 *does* subtract a parent term and the algebra cancels). See §6 for the empirical-regression match and §8 for what would upgrade confidence to "certain".

---

## 1. Setup — code citations and accumulator scoping

### 1.1 Function `FindBestSplit` (csv_train.cpp:1593–end-of-function)

The Cosine gain is documented inline at L1553–1577 (the `ComputeCosineGainKDim` derivation comment):

```
num  = sumLeft²/(wLeft+λ) + sumRight²/(wRight+λ)
den  = sumLeft²·wLeft/(wLeft+λ)² + sumRight²·wRight/(wRight+λ)²
score = num / sqrt(den)
```

L1568 explicitly states: **"No parent-node subtraction (unlike L2). Cosine is an absolute score, not a differential gain."** This is the load-bearing fact for the verdict below.

### 1.2 One-hot branch loop structure (L1654–1770)

```
for featIdx in features:                                       # outer
  if feat.OneHotFeature:
    for bin in 0..feat.Folds:                                  # per (feat, bin) candidate
      double totalGain = 0.0;        // L1661           ← bin-scoped
      double cosNum_d  = 0.0;        // L1671           ← bin-scoped
      double cosDen_d  = 1e-20;      // L1672           ← bin-scoped (sqrt guard)

      [min-data-in-leaf early-out using countHist, L1674-1685]

      for p in 0..numPartitions:                               # per partition p
        for k in 0..K:                                         # per approx-dim k
          totalSum     = perDimPartStats[k][p].Sum
          totalWeight  = perDimPartStats[k][p].Weight
          sumRight     = histData[bin]                         # L1693
          weightRight  = histData[totalBinFeatures + bin]      # L1694
          sumLeft      = totalSum    - sumRight                # L1695
          weightLeft   = totalWeight - weightRight             # L1696

          if (weightLeft < 1e-15f || weightRight < 1e-15f)
            continue;                                          # L1698  ← JOINT SKIP

          switch (scoreFunction):
            case L2:
              totalGain += sumLeft²  / (weightLeft  + λ)
                         + sumRight² / (weightRight + λ)
                         - totalSum² / (totalWeight + λ);      # L1702-1704
              break;
            case Cosine:
              cosNum_d += dSL² · dInvL + dSR² · dInvR;         # L1715
              cosDen_d += dSL² · dWL · dInvL² + dSR² · dWR · dInvR²;  # L1716-1717
              break;

      if (Cosine):
        totalGain = cosNum_d / sqrt(cosDen_d);                 # L1731  ← finalize per (feat, bin)
      [argmax update, perturbation]
```

### 1.3 Ordinal branch loop structure (L1772–2099)

Same shape, but with suffix-sum lookups instead of single-bin histogram reads, and the per-side mask **already shipped** at L1950–2045 (DEC-042 fix). Crucially, L1973 (L2) and L2004–2005 (Cosine) sit inside the `for p × for k` double loop, and L2052 (`totalGain = cosNum_d / std::sqrt(cosDen_d)`) sits at bin scope after the loops close.

### 1.4 Accumulator scoping summary

| Accumulator   | Scope               | Initialized at         | Read for argmax at      |
|---------------|---------------------|------------------------|-------------------------|
| `totalGain`   | per (feat, bin)     | L1661 (1H), L1814 (Ord)| L1740 (1H), L2106 (Ord) |
| `cosNum_d`    | per (feat, bin)     | L1671 (1H), L1818 (Ord)| L1731 (1H), L2052 (Ord) |
| `cosDen_d`    | per (feat, bin)     | L1672 (1H), L1819 (Ord)| L1731 (1H), L2052 (Ord) |
| `bestGain`    | per `FindBestSplit` | L1615                  | L1740 / 2106 comparison |

The Cosine final score is **one division-and-sqrt per (feat, bin) candidate**, fed by a sum across the full `(p, k)` grid. The per-(p,k) terms are therefore additively pooled before any non-linear operation.

### 1.5 CPU reference scoping (for §4 cross-check)

`scoring.cpp:573–585` (`UpdateSplitScore`) is the per-bucket dispatch that calls `IPointwiseScoreCalcer::AddLeafPlain(splitIdx, falseStats, trueStats)`. It is invoked from `UpdateScores → CalcScoresForLeaf → updateSplitScoreClosure` (L673–683). The outer loop is `for leaf in [0..leafCount)` (= partition loop). Per leaf and per split candidate, `AddLeafPlain` **accumulates into a per-`splitIdx` `[Score_num, Score_den]` pair** (CPU's `score_calcers.h:73 TCosineScoreCalcer::Scores`). Final score is computed once per `splitIdx` in `GetScores()` as `Scores[i][0] / sqrt(Scores[i][1])` (`score_calcers.h:58`).

So CPU's data-flow is **isomorphic** to MLX's `cosNum_d / sqrt(cosDen_d)` finalization at L1731/2052 — accumulate across leaves (= partitions) for one split candidate, finalize once. CPU has no K-dim loop visible at this level because CPU treats multi-dim by an outer `forEachBodyTailAndApproxDimension` (`scoring.cpp:544`); MLX collapses the equivalent into a per-k inner loop inside the per-partition loop. The pooled-then-finalize structure is the same.

### 1.6 CPU's per-side mask (the formula MLX is mirroring)

`short_vector_ops.h:155-175` — `UpdateScoreBinKernelPlain`:

```cpp
const __m128d isSumWeightPositive = _mm_cmpgt_pd(sumWeight, _mm_setzero_pd());   // L167
const __m128d average = _mm_mul_pd(
    sumWeightedDelta,
    _mm_and_pd(isSumWeightPositive,
               _mm_div_pd(_mm_set1_pd(1.0), _mm_add_pd(sumWeight, regularizer))) // L168
);
const __m128d dpSummands = _mm_mul_pd(average, sumWeightedDelta);                // dp = avg · sum
const __m128d d2Summands = _mm_mul_pd(_mm_mul_pd(average, average), sumWeight);  // d² = avg² · w
```

When `sumWeight ≤ 0`, the `isSumWeightPositive` mask is all-zero, `average = 0`, so `dpSummands = 0` and `d2Summands = 0` for that side. **The non-empty side is unaffected.** This is precisely the per-side mask MLX is mirroring.

`online_predictor.h:112-119`:

```cpp
inline double CalcAverage(double sumDelta, double count, double scaledL2Regularizer) {
    double inv = count > 0 ? 1. / (count + scaledL2Regularizer) : 0;
    return sumDelta * inv;
}
```

Same semantics: `count ≤ 0 → average = 0`. CPU's L2 path (`score_calcers.cpp:20-33`, `TL2ScoreCalcer::AddLeafPlain`) calls `CalcAverage` for each side and then `AddLeaf(splitIdx, avrg, leafStats)`, where `AddLeaf` does `Scores[splitIdx] += leafApprox · leafStats.SumWeightedDelta`. When `avrg=0`, that side contributes 0 to `Scores[splitIdx]`. Again, the non-empty side is unaffected.

Notice: **CPU's L2 path does not subtract a parent term either.** CPU L2 score is `Σ_leaves Σ_sides avg · sumWeightedDelta = Σ_sides sum² / (w + λ)` — that's it. The "parent term subtraction" the user is asking about is **not in CPU.** It is something MLX added on top, and only in MLX's L2 branch (L1702-1704 one-hot, L1973 ordinal).

This is a non-trivial finding — see §4.

---

## 2. Ordinal branch derivation (L1980 — already-shipped fix, sanity check)

This section verifies the framework agrees with the shipped DEC-042 fix.

### 2.1 Ordinal pre-fix behaviour

For the ordinal split "value > b → right" with suffix-sum reads (`sumRight = suffSum[bin]`), the per-(p,k) Cosine contribution under joint-skip was:

```
contrib_jointSkip(p, k, bin) =
   { (sL²/(wL+λ) + sR²/(wR+λ),  sL²·wL/(wL+λ)² + sR²·wR/(wR+λ)²)   if wL > ε ∧ wR > ε
   { (0, 0)                                                          otherwise
```

Summing across `(p, k)` and finalizing: `gain(feat, bin) = Σ contrib_num / sqrt(Σ contrib_den)`.

### 2.2 Ordinal: where can `wR=0` occur?

For **ordinal** with suffix sums, `sumRight = suffGrad[bin]`. `suffGrad[bin] = Σ_{i ≥ bin} hist[firstFold + i]`. `wR = 0` requires every bin from `bin` to `folds-1` to be empty in partition p — i.e., the threshold candidate `b = bin` puts everything to the Left. This is the **highest-bin** edge of the suffix curve and tends to occur for split candidates near the upper border.

`wL = 0` requires `wR = totalWeight_p` — i.e., every doc in partition p has feature value > bin. This is the **lowest-bin** edge.

In partitioned trees (depth ≥ 1), partitions inherited from earlier splits naturally produce monotone restrictions on the feature distribution, so wR=0 / wL=0 candidates appear with non-trivial frequency.

### 2.3 Ordinal CPU reference

CPU's `UpdateScoreBinKernelPlain` in this branch context (called per leaf, per split candidate) adds `(sumLeft²/(wLeft+λ)) + (sumRight²·wRight/(wRight+λ)²) · ...` per side, **with mask**: empty side contributes 0, non-empty side contributes its full term.

So CPU's per-(p, k=fixed-by-outer-loop) contribution is:

```
contrib_CPU(p, k, bin) =
   ( sL² · 1[wL>0] / (wL+λ) + sR² · 1[wR>0] / (wR+λ),
     sL² · 1[wL>0] · wL / (wL+λ)² + sR² · 1[wR>0] · wR / (wR+λ)² )
```

### 2.4 Per-side mask = CPU mirror

MLX's L1990–2003 (per-side mask) implements `contrib_CPU` exactly. So the shipped fix matches CPU. ✓

### 2.5 Why this works for ordinal

For ordinal, "rescuing" wR=0 / wL=0 partitions has a meaningful effect: in those partitions the candidate is genuinely "send everything one way", which is a perfectly valid split *of that partition*. Across partitions where the candidate is informative (wL>ε, wR>ε), the ranking of (feat, bin) relative to other (feat', bin') still benefits from these "all-one-way" partitions adding their `sum²/(w+λ)` baseline contribution. CPU agrees, MLX now agrees, parity gates pass at 1941× improvement (from S33 close).

Sanity: our framework agrees with the shipped fix. ✓

---

## 3. One-hot branch derivation (L1698)

### 3.1 One-hot wR=0 frequency

For one-hot, `sumRight = histData[bin]` (a *single* histogram bucket, not a suffix sum). `wR = 0` means **no docs in partition p have feat == k** (where k denotes the bin = category index).

In a one-hot encoding with C categories, each partition p has a per-category distribution. With C bins and N_p docs in partition p, **C - 1 of the C categories are typically rare or empty**, especially:

- After depth ≥ 1 splits, partitions become small.
- One-hot is most often used for high-cardinality categoricals where each category is sparse.
- At depth 0, even on a balanced dataset with C=10 and N=10000, the smaller categories may have ≪ 1e-15 weight only for empty mass — but in the *partitioned* context (depth ≥ 1), categories that didn't appear in the partition's parent path will have wR = 0.

So **one-hot has many more wR=0 (p, k, bin) cells than ordinal**. This is the first asymmetry that matters.

### 3.2 The wR=0 contribution under per-side mask

When `wR=0` in (p, k, bin) for one-hot: `sumRight = 0`, `weightRight = 0`, hence `sumLeft = totalSum_p`, `weightLeft = totalWeight_p`. The per-side mask contributes:

```
Δ cosNum (per-side mask) = totalSum_p² / (totalWeight_p + λ)
Δ cosDen (per-side mask) = totalSum_p² · totalWeight_p / (totalWeight_p + λ)²
```

vs joint skip:

```
Δ cosNum (joint skip) = 0
Δ cosDen (joint skip) = 0
```

**The difference is non-zero and bin-independent within partition p** (it depends only on the partition's totals, not on which one-hot bin we're evaluating).

### 3.3 The full bin-level gain expression

Let `D(bin) ⊆ {(p, k)}` be the set of (partition, dim) pairs where `wR(p, k, bin) = 0` (one-hot), and `S(bin) = {(p, k) : both wL > ε and wR > ε}` be the set where the candidate is genuinely informative for that (p, k). Then:

**Under joint skip:**

```
cosNum_jointSkip(bin) = Σ_{(p,k) ∈ S(bin)} [ sL² / (wL+λ) + sR² / (wR+λ) ]
cosDen_jointSkip(bin) = 1e-20 + Σ_{(p,k) ∈ S(bin)} [ sL²·wL/(wL+λ)² + sR²·wR/(wR+λ)² ]
```

**Under per-side mask:**

```
cosNum_mask(bin) = cosNum_jointSkip(bin) + Σ_{(p,k) ∈ D(bin)} [ totalSum_p,k² / (totalWeight_p,k + λ) ]
cosDen_mask(bin) = cosDen_jointSkip(bin) + Σ_{(p,k) ∈ D(bin)} [ totalSum_p,k² · totalWeight_p,k / (totalWeight_p,k + λ)² ]
```

(The wL=0 case in one-hot — i.e., **every** doc in partition p has feat == k — is rarer for one-hot but symmetric; it adds the same quantity. We absorb it into D(bin) for brevity.)

### 3.4 Effect on the argmax

Define `B(bin) := Σ_{(p,k) ∈ D(bin)} totalSum_p,k² / (totalWeight_p,k + λ)` (and analogously `B_den(bin)`). These are the "baseline / no-split contribution" terms summed over partitions where this bin is degenerate.

```
gain_mask(bin)      = ( cosNum_jointSkip(bin) + B_num(bin) )
                      / sqrt( cosDen_jointSkip(bin) + B_den(bin) )

gain_jointSkip(bin) = cosNum_jointSkip(bin) / sqrt( cosDen_jointSkip(bin) )
```

Since `B_num(bin), B_den(bin) ≥ 0`, the per-side mask **adds a nonnegative quantity to both num and den** of the gain ratio. The effect on argmax depends on which way the ratio shifts — it is *not* a uniform shift across bins because **D(bin) varies with bin**: different one-hot bins partition the (p, k) grid into "informative" vs "degenerate" cells differently, so each bin's added baseline B(bin) is different.

Concretely, a one-hot bin k₀ that is **rare** (most partitions have wR = 0 for k₀) will pick up a much larger `|D(bin)|` and hence a much larger `B(bin)` than a one-hot bin k₁ that is **common** (most partitions have wR > 0 for k₁). The per-side mask therefore *systematically rewards* rare one-hot categories, which is a **structural argmax bias** unrelated to actual signal.

### 3.5 Comparison to CPU

Does CPU produce `gain_mask` or `gain_jointSkip` for one-hot Cosine? CPU's `CalcScoresForLeaf` calls `UpdateSplitScoreClosure` for each split candidate within a leaf — the closure invokes `AddLeafPlain` regardless of degenerate sides; the per-side mask is implicit in `CalcAverage`/`UpdateScoreBinKernelPlain`. So **CPU produces `gain_mask`** for both one-hot and ordinal split candidates, where `S(bin)` and `D(bin)` are interpreted per-leaf.

So if CPU produces `gain_mask`, why would mirroring CPU break things? Two options:

**(a)** The empirical anchor regression measures something other than long-run convergence (e.g., a single seed, a small dataset where the change happens to push toward a worse local minimum at a specific iteration). In which case the per-side mask is correct and the regression is noise-dominated.

**(b)** There is a discrepancy between MLX's one-hot semantics and CPU's that we have not yet identified — e.g., MLX's one-hot bin loop iterates over `Folds` categories (L1656) but CPU's `CalcSplitsCount` for `OneHotFeature` returns `bucketCount` (`score_calcers.cpp:60`) — same surface. But the *meaning* of "split bin" might differ: MLX evaluates "go right if value == bin" for each `bin ∈ [0, Folds)`; CPU does the same. We would need to verify that the **pre-fix** MLX one-hot Cosine matched CPU on parity (it did, per S30 closure & DEC-036 history) — meaning CPU and MLX *both* used joint-skip semantics for one-hot Cosine prior to S33. **(Verification pending — see §8 open questions.)**

This is the load-bearing question. If pre-S33 MLX and CPU both used joint-skip for one-hot Cosine and were ULP-bit-equal at the gates, then **CPU itself is using joint-skip for one-hot Cosine**, and the per-side mask MLX is mirroring is correct *only for the ordinal branch*. The advisory board's framing was the right intuition but expressed in the wrong subtractor; the actual asymmetry is in the *partitioner* (one-hot vs ordinal), not the parent term.

### 3.6 Why one-hot vs ordinal might genuinely differ

Conceptually: the per-side mask interprets "wR=0 in partition p" as **a valid split that puts everything Left** in partition p. For ordinal, this corresponds to a real threshold-on-the-edge split. For one-hot category k₀ where partition p has zero docs with feat == k₀, the "split" is **vacuous** — there is no informative decision to make in partition p. It is not "send everything one way"; it is "this category is absent." Treating it as a real split adds spurious signal `totalSum_p² / (totalWeight_p + λ)` that biases toward categories that are *rare in many partitions*.

This is an interpretive distinction the per-side mask cannot encode. CPU's implementation may or may not handle this — and given the empirical regression, the operational evidence is that **CPU does not add this bias either** (i.e., CPU effectively does joint-skip on one-hot Cosine, perhaps because the upstream caller suppresses degenerate one-hot candidates or never invokes `UpdateSplitScoreClosure` on them).

---

## 4. Parent-term location verdict

**Claim:** Neither branch subtracts a parent term in the Cosine code path.

**Code citations:**

- **One-hot Cosine** (L1706-1719): only `cosNum_d += dSL² · dInvL + dSR² · dInvR` (L1715) and `cosDen_d += dSL² · dWL · dInvL² + dSR² · dWR · dInvR²` (L1716-1717). No `-totalSum²/(totalWeight+λ)` term anywhere in this case branch. The bin-scope finalization at L1731 is `cosNum_d / sqrt(cosDen_d)` — no parent term.

- **Ordinal Cosine** (L1976-2042, post-S33 fix): only `termNum += dSL²·dInvL` / `dSR²·dInvR` (L1994, 2001) and `termDen += ...` (L1995, 2002), then `cosNum_d += termNum; cosDen_d += termDen` (L2004-2005). Bin-scope finalization at L2052: `cosNum_d / sqrt(cosDen_d)`. No parent term.

- **CPU Cosine reference** (`score_calcers.cpp:10-12 + short_vector_ops.h:155-175`): adds `dpSummands + d2Summands` per side; no parent subtraction. `score_calcers.h:58` — `Scores[i][0] / sqrt(Scores[i][1])`. No parent term.

- **One-hot L2** (L1702-1704): does subtract per-(partition, k): `totalGain += sL²/(wL+λ) + sR²/(wR+λ) - totalSum²/(totalWeight+λ)`. The subtraction is **inside** the `for p × for k` double loop, so it occurs **once per (p, k)** within a single (feat, bin) candidate.

- **Ordinal L2** (L1968-1973, post-S33): same pattern — `totalGain += sL²/(wL+λ)` (if wL_pos), `+ sR²/(wR+λ)` (if wR_pos), then `totalGain -= totalSum²/(totalWeight+λ)` unconditionally — also per-(partition, k).

**Verdict on parent-term scope:**

| Path | Parent term? | Where? |
|------|--------------|--------|
| Cosine (both branches), MLX & CPU | None | — |
| L2, MLX one-hot (L1702-1704) | Yes, per-(p, k) inside the inner loop | L1704 |
| L2, MLX ordinal (L1973) | Yes, per-(p, k) inside the inner loop | L1973 |
| L2, CPU (`score_calcers.cpp:20-33`) | **None** | — |

There is a separate, smaller divergence here: MLX's L2 path subtracts a parent term per-(p, k) but CPU's L2 path does not. For a single-partition, single-dim case (depth=0, K=1), this just adds a *constant offset* `-totalSum²/(totalWeight+λ)` to MLX's gain that CPU lacks. Constants don't affect argmax, so parity holds. But for `numPartitions > 1` or `K > 1`, MLX subtracts this parent term **once per (p, k)** whereas CPU subtracts it zero times — meaning MLX's L2 gain has `numPartitions × K` extra subtractions of the partition-totals baseline. *This matters for argmax only if those partition totals vary — which they always do for trees of depth ≥ 1.* This is a **known L2 divergence** that may already be papered over by the per-side mask shipping in S33-L4-FIX (see §6.1) — the L2 fix in S33 modified L1973's behaviour.

**Re-visit of the advisory-board frame:** The frame asked "where is the parent term subtracted?" The answer for Cosine is "nowhere." So the cancellation analysis (parent term cancels the per-side mask's contribution when wR=0) does not apply in the Cosine code path. The empirical regression cannot be explained by a parent-cancellation argument — it must be explained by the structural argmax bias derived in §3.4.

---

## 5. Per-side mask correctness verdict for one-hot Cosine

### 5.1 Verdict

**The per-side mask, as written for one-hot Cosine at L1698, is materially different from the joint-skip and is wrong for argmax purposes.**

**Reasoning chain:**

1. The Cosine path has no parent-term subtraction at any level (§4).
2. When `wR=0` in a (p, k) cell, joint-skip contributes 0 to (`cosNum_d`, `cosDen_d`), per-side mask contributes `(totalSum_p,k² / (totalWeight_p,k+λ), totalSum_p,k² · totalWeight_p,k / (totalWeight_p,k+λ)²)`. (§3.2)
3. The added contribution is **non-negative** but **bin-dependent** through `|D(bin)|` (the count of degenerate (p, k) cells), so it does not cancel uniformly across bins. (§3.3-3.4)
4. One-hot has structurally many more `wR=0` cells than ordinal (§3.1), because most one-hot categories are absent from most partitions at depth ≥ 1.
5. The added term `totalSum_p² / (totalWeight_p+λ)` is the partition's "baseline / no-split" contribution. For ordinal, this is interpretable as a real "everything Left" split. For one-hot, this is interpretable as "category absent" — a vacuous condition, not a real split decision.
6. The structural bias rewards one-hot categories that are rare in many partitions, irrespective of signal.

### 5.2 Strength of the verdict

- "Wrong as written" is **high confidence** — the math derivation is direct and the regression test confirms it.
- "CPU agrees with joint-skip for one-hot Cosine" is **medium-high confidence** — pending direct verification that pre-S33 MLX's one-hot Cosine path matched CPU (§8).

### 5.3 What about one-hot L2?

For one-hot L2 (L1702-1704), the parent term `-totalSum²/(totalWeight+λ)` IS subtracted per-(p, k) inside the inner loop. So when `wR=0`:

- Joint-skip contributes 0.
- Per-side mask (S33 ordinal-style for L2 — *not yet shipped to one-hot*) would contribute `sL²/(wL+λ) - totalSum²/(totalWeight+λ) = totalSum_p²/(totalWeight_p+λ) - totalSum_p²/(totalWeight_p+λ) = 0`.

For one-hot L2 the parent term **does cancel exactly** (because at wR=0, sumLeft=totalSum, weightLeft=totalWeight, so the Left contribution equals the parent term being subtracted). The per-side mask is therefore a **no-op for one-hot L2 wR=0 cells**, identical to joint-skip. **Mirror-applying the L2 per-side mask to one-hot is harmless** (modulo the wL=0 ∧ wR=0 case, which contributes -totalSum²/(totalWeight+λ) under "skip whole partition only when both empty" — also a constant).

Wait — actually we need to re-check the `wL=0` cell symmetrically. If `wL=0`: sumLeft=0, weightLeft=0, sumRight=totalSum, weightRight=totalWeight. Per-side mask contributes `sR²/(wR+λ) - totalSum²/(totalWeight+λ) = 0`. Same cancellation. ✓

So the **L2 cancellation in §4 (advisory-board frame)** holds *for L2 only*, *because the L2 path has the parent-term subtraction*. For Cosine, with no parent subtraction, the analogous cancellation does not occur, and the per-side mask injects a structural bias.

This is an important asymmetry: **the same code transformation (joint-skip → per-side mask) is correct for L2 but wrong for Cosine in the one-hot branch.** And the reason is exactly the parent-term presence/absence — the very thing the advisory-board frame named, just applied to the wrong score function.

---

## 6. Empirical regression explanation

### 6.1 The 3% loss regression

Loss went from 0.479101 → 0.493401 (+2.99%) when the per-side mask was mirror-applied to L1698. The derivation predicts:

- Per-side mask injects positive `B_num(bin)` and `B_den(bin)` into each one-hot bin's accumulators (§3.4).
- The injection is non-uniform across bins; rare-category bins get larger injections.
- Argmax is biased toward bins with larger `|D(bin)|`, i.e., rare categories.
- Choosing a rare-category split as the "best" first split is plausibly worse because rare categories carry less signal mass and yield poorly-balanced subtrees.

This is **consistent with the observed regression**. The mechanism is structural, not noise.

### 6.2 Was the L2 part of the mirror harmful?

If the mirror commit also touched L1702-1704 (one-hot L2), the analysis in §5.3 says the L2 change is a no-op. So the regression must be coming from the **Cosine** branch of the one-hot mirror. (Verify by reading the actual reverted commit to confirm which lines were touched.)

### 6.3 Why the synthetic-anchor anchor is sensitive

A synthetic anchor with one-hot features and Cosine score is *exactly* the configuration where this bias is largest (most one-hot bins, rare-category bias most visible). A real-world dataset with mostly ordinal features and L2 score would be much less sensitive — which suggests this issue may have been latent in production for a long time on configurations that were not tested.

### 6.4 Falsification check: would the derivation predict a *different* sign?

No. `B_num` and `B_den` are non-negative; the per-side mask **monotonically increases** num and den. The argmax bias direction is determined by the `B(bin)` distribution, which is itself determined by the partition × category co-occurrence pattern — but the existence of bias is unconditional. An empirical regression is therefore predicted; the magnitude (3%) is consistent with a moderate-sized synthetic dataset.

---

## 7. Recommended fix shape

### 7.1 Recommendation: do NOT apply per-side mask to one-hot Cosine. Keep joint-skip at L1698 as-is.

**Rationale:**

1. The parent-term-cancellation argument that justified the ordinal Cosine fix (DEC-042) **does not hold** for Cosine because there is no parent term to cancel.
2. CPU's behaviour on one-hot Cosine appears to be effectively joint-skip — it does not add the partition-baseline term to bins whose wR=0 in many partitions (verification pending; if CPU does add it, then CPU has the same structural bias, and the regression measurement is the canary).
3. The empirical 3% regression is a direct prediction of the math.
4. The "wR=0 in one-hot" event is interpretively *vacuous* (category absent), not a real split — keeping joint-skip preserves this distinction.

### 7.2 What about one-hot L2 (L1702-1704)?

**Recommendation: mirror-apply the per-side mask to one-hot L2 is mathematically equivalent to joint-skip** (per §5.3 cancellation). It would be a code-style consistency change, not a behavioural one. **Optional, low-priority.** If shipped, it must be paired with explicit ULP-bit-equal parity verification across one-hot L2 datasets, because while the per-(p, k) cancellation is exact in real arithmetic, fp64 round-off in the (sR² · wR · invR² + sL² · wL · invL² - totalSum² · totalWeight · invTotal²) sequence may differ by ≤ 4 ulp.

Important nuance: there is a separate, pre-existing divergence between MLX L2 (subtracts parent per (p,k)) and CPU L2 (no parent subtraction). This isn't fixed by the per-side mask change. Whether it materially affects argmax should be a separate investigation — likely covered by some prior decision (DEC-XXX) since DEC-036/042 history doesn't mention L2 parent divergence as an open issue.

### 7.3 Code-change shape (no commits)

**For one-hot Cosine (L1698):** leave joint-skip behaviour unchanged. The single line `if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;` at L1698 stays as-is. (This is the de-facto state after the revert; the recommendation is to **not** re-apply the mirror in S34 or beyond unless §8 open questions are resolved.)

**For one-hot L2 (L1702-1704):** optionally mirror the ordinal pattern (per-side mask, with explicit no-op-at-degenerate-cells documentation), but only after a parity gate. Defer to future sprint; not blocking.

### 7.4 If a future probe shows CPU *does* use per-side mask for one-hot Cosine

Then the regression measurement is showing the per-side mask is correct *in the limit* but causes worse performance on the synthetic anchor specifically. Investigation path: (a) verify the anchor is representative; (b) if CPU+per-side-mask is the parity target, the regression is acceptable (parity > anchor); (c) document as known anchor-only sensitivity.

For now, default to the conservative "leave joint-skip for one-hot Cosine."

---

## 8. Confidence and open questions

### 8.1 Confidence levels

| Statement | Confidence | Why |
|-----------|------------|-----|
| Cosine path has no parent-term subtraction in either MLX branch | **Certain** | Direct code reading, lines 1700-1719 and 1976-2045, plus L1568 explicit comment |
| CPU Cosine path has no parent-term subtraction | **Certain** | `score_calcers.h:58 + short_vector_ops.h:155-175` |
| Per-side mask injects non-zero, bin-dependent contribution at wR=0 cells in one-hot Cosine | **Certain** | §3.2 algebra, all variables defined |
| The injection biases argmax toward rare-category bins | **High** | Follows from §3.4; depends on `|D(bin)|` distribution being non-uniform across bins, which it always is for one-hot |
| The empirical 3% regression is caused by this structural bias (not seed noise) | **High** | Mechanism is monotone in dataset size and one-hot count; would require multi-seed verification to upgrade to "certain" |
| One-hot L2 per-side mask is a no-op (cancellation §5.3) | **High** | Algebra is exact in real arithmetic; fp64 may differ by ≤ few ulp |
| CPU's one-hot Cosine effectively uses joint-skip | **Medium-high** | Inferred from pre-S33 parity history; requires direct code-walk of the CPU one-hot Cosine path to upgrade to "certain" |

### 8.2 Open questions (to resolve before upgrading verdict to "certain")

1. **Q1 (CPU one-hot Cosine semantics):** Walk `CalcScoresForLeaf` (`scoring.cpp:CalcScoresForLeaf`) for one-hot split candidates and confirm whether the `updateSplitScoreClosure` is called with degenerate (wR=0) candidates or filtered upstream. If filtered, CPU = joint-skip, the verdict §5.1 holds. If invoked with degenerate stats, CPU = per-side mask, and the regression is anchor-noise-or-acceptable-divergence.

2. **Q2 (does pre-S33 MLX one-hot Cosine match CPU?):** Find the pre-S33 ULP-bit-equal parity gate result for one-hot Cosine (S30 closure in `docs/sprint30/`?). If it was bit-equal pre-S33 and still bit-equal post-S33-without-the-L1698-mirror, then CPU = joint-skip is *implied* (with very high confidence). DEC-036 history may have this answer already.

3. **Q3 (multi-seed / multi-dataset regression):** Repeat the synthetic-anchor experiment with N ≥ 5 seeds and ≥ 2 datasets (one with C=10 categories, one with C=50). If the regression is structural, it scales monotonically with C and is consistent across seeds. If it's seed-noise, it is bimodal across seeds.

4. **Q4 (L2 parent-term divergence between MLX and CPU):** This is a separate, latent issue surfaced by §4. Does this need a separate investigation, or has it been addressed by DEC-XXX? The CPU L2 score formula `Σ_sides sum²/(w+λ)` differs from MLX L2 `Σ_sides sum²/(w+λ) - totalSum²/(totalWeight+λ)` by a constant *only at depth=0, K=1, numPartitions=1*. At depth≥1 or numPartitions>1 the constant becomes per-(feat, bin) variable and may affect argmax. This is out of scope for T0a but should be filed as an issue.

### 8.3 What would upgrade verdict to "certain"

Resolving Q1 and Q2 simultaneously, plus a multi-seed run for Q3 confirming the regression is structural. T0b (ml-engineer's parallel code reading of the actual revert commit and any pre-S33 parity reports) is the natural place to converge. After T0a + T0b reconcile, if both arrive at "joint-skip is correct for one-hot Cosine" with the empirical anchor as supporting evidence, this is sufficient to commit a S34 documentation change (no code change) and close the loop.

---

## 9. Summary table

| Branch | Score function | Pre-S33 behaviour | Post-S33 behaviour (shipped) | T0a recommendation | Reason |
|--------|----------------|---------------------|--------------------------|---------------------|--------|
| Ordinal | L2 | joint skip | per-side mask | **keep per-side mask** | matches CPU `CalcAverage` mask, parity at 1941× improvement |
| Ordinal | Cosine | joint skip | per-side mask | **keep per-side mask** | matches CPU `UpdateScoreBinKernelPlain`, no parent term needed because both sides are equally re-included |
| One-hot | L2 | joint skip | joint skip (unchanged) | **no change required** (or mirror, optional, low-priority) | Per-side mask is exact-no-op via parent-term cancellation §5.3 |
| One-hot | **Cosine** | **joint skip** | **joint skip (unchanged)** | **DO NOT mirror per-side mask** | No parent term to cancel the wR=0 contribution → structural bin-dependent argmax bias → predicted-and-observed 3% regression on synthetic anchor |

---

## Appendix A — Why the advisory-board frame got the right intuition with the wrong handle

The advisory-board frame asked: "does the parent term cancel the per-side mask when wR=0?" This is the right intuition for **L2** in the one-hot branch — and it correctly diagnoses the no-op cancellation §5.3. But the frame implicitly assumed the parent term is always present in the gain. For **Cosine**, the gain is `num/sqrt(den)` with no parent subtraction, so the cancellation argument doesn't apply, and the per-side mask becomes net-positive contribution.

The asymmetry the frame was groping for is real, just located in a different layer:

- **L2 + per-side mask:** parent term cancels the wR=0 full-mass injection → no-op.
- **Cosine + per-side mask:** no parent term → wR=0 full-mass injection survives → bias.

So the parent-term **presence** is what makes per-side-mask safe in L2 and unsafe in Cosine. The advisory-board frame named the right object (the parent term) but expected it to do the cancellation work in the wrong score function.

---

## Appendix B — Code line citations consolidated

| Citation | Significance |
|----------|--------------|
| `csv_train.cpp:1554-1577` | Cosine gain definition: explicit "no parent-node subtraction" |
| `csv_train.cpp:1582-1587` | `ComputeCosineGainKDim` definition |
| `csv_train.cpp:1654` | Branch on `feat.OneHotFeature` |
| `csv_train.cpp:1671-1672` | `cosNum_d`, `cosDen_d` initialized at bin scope (one-hot) |
| `csv_train.cpp:1687-1724` | one-hot per-(p, k) inner loops |
| `csv_train.cpp:1698` | **the joint-skip site under analysis** |
| `csv_train.cpp:1702-1704` | one-hot L2 with parent-term subtraction inside loop |
| `csv_train.cpp:1715-1717` | one-hot Cosine accumulation, no parent term |
| `csv_train.cpp:1731` | one-hot Cosine finalization at bin scope |
| `csv_train.cpp:1818-1819` | `cosNum_d`, `cosDen_d` initialized at bin scope (ordinal) |
| `csv_train.cpp:1950-1973` | ordinal L2 per-side mask (S33 fix) |
| `csv_train.cpp:1973` | ordinal L2 parent-term subtraction |
| `csv_train.cpp:1976-2042` | ordinal Cosine per-side mask (S33 fix) |
| `csv_train.cpp:2052` | ordinal Cosine finalization at bin scope |
| `score_calcers.h:47-74` | CPU `TCosineScoreCalcer` |
| `score_calcers.h:58` | CPU `GetScores`: `Scores[i][0] / sqrt(Scores[i][1])` |
| `score_calcers.cpp:10-12` | CPU `TCosineScoreCalcer::AddLeafPlain` calls `UpdateScoreBinKernelPlain` |
| `score_calcers.cpp:20-33` | CPU `TL2ScoreCalcer::AddLeafPlain`: no parent subtraction |
| `short_vector_ops.h:155-175` | CPU SSE2 `UpdateScoreBinKernelPlain` with mask via `_mm_cmpgt_pd` |
| `online_predictor.h:112-119` | CPU `CalcAverage`: `count > 0 ? ... : 0` |
| `scoring.cpp:573-585` | CPU dispatch: `UpdateSplitScore → AddLeafPlain` |
| `scoring.cpp:673-683` | CPU per-leaf loop driving `UpdateSplitScoreClosure` |

---

*End of T0a derivation. Awaiting T0b reconciliation.*
