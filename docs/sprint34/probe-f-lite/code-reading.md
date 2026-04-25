# S34-PROBE-F-LITE T0b: Code Reading Report

**Date**: 2026-04-25
**Branch**: mlx/sprint-34-probe-f-lite (read-only investigation)
**Kernel md5 invariant**: 9edaef45b99b9db3e2717da93800e76f (not modified)
**File under investigation**: `catboost/mlx/tests/csv_train.cpp`

---

## 1. Loop Nesting Map

### One-hot branch (lines 1654–1770)

```
for featIdx in features:                              # L1649
    if feat.OneHotFeature:                            # L1654
        for bin in 0..feat.Folds:                     # L1656
            totalGain = 0.0           [bin scope]     # L1661
            cosNum_d  = 0.0           [bin scope]     # L1671
            cosDen_d  = 1e-20         [bin scope]     # L1672
            for p in 0..numPartitions:                # L1687
                for k in 0..K:                        # L1688
                    [compute sums]
                    if (wL < 1e-15 || wR < 1e-15) continue;  # L1698
                    switch scoreFunction:
                        L2:  totalGain += sL²/(wL+λ) + sR²/(wR+λ) - sP²/(wP+λ)
                        Cosine: cosNum_d += ...; cosDen_d += ...
            # after (p,k) double loop:
            if Cosine: totalGain = cosNum_d / sqrt(cosDen_d)   # L1731
            argmax update                                        # L1740
```

Loop structure:
- `bin` is the outermost candidate loop (L1656–L1770)
- `p` (partition) is inside `bin` (L1687–L1724)
- `k` (approx dimension) is innermost (L1688–L1723)
- `totalGain`, `cosNum_d`, `cosDen_d` are **declared and initialized at `bin` scope** (L1661–1672)
- Cosine finalization is **outside both `p` and `k` loops** (L1726–1732)
- The `continue` at L1698 skips the rest of the `k` body for one `(p,k)` pair

### Ordinal branch (lines 1771–2139)

```
for featIdx in features:                              # L1649
    else (ordinal):                                   # L1771
        [build suffix sums: L1783-1793]
        for bin in 0..folds:                          # L1812
            totalGain = 0.0           [bin scope]     # L1814
            cosNum_d  = 0.0           [bin scope]     # L1818
            cosDen_d  = 1e-20         [bin scope]     # L1819
            for p in 0..numPartitions:                # L1856
                for k in 0..K:                        # L1878
                    [compute sums from suffix arrays]
                    const wL_pos = (wL > 1e-15)       # L1950  [S33-L4-FIX]
                    const wR_pos = (wR > 1e-15)       # L1951  [S33-L4-FIX]
                    switch scoreFunction:
                        L2: per-side mask + parent subtraction  # L1954-1974
                        Cosine: per-side mask, no parent term   # L1976-2005
            # after (p,k) double loop:
            if Cosine: totalGain = cosNum_d / sqrt(cosDen_d)    # L2052
            argmax update                                         # L2108
```

Key difference from one-hot: the `continue` at L1698 (one-hot) was the **unfixed** joint skip. The ordinal branch received the S33-L4-FIX (Commit 1.5, per-side mask at L1950–2005). The one-hot branch retains the old `continue` at L1698.

---

## 2. Accumulator Scope

### One-hot branch

| Variable | Declared at | Scope | Reset cadence |
|----------|-------------|-------|---------------|
| `totalGain` | L1661 | `bin` block | Per candidate bin |
| `cosNum_d` | L1671 | `bin` block | Per candidate bin |
| `cosDen_d` | L1672 | `bin` block | Per candidate bin |

All three are declared inside the `for bin` loop body, immediately before the `(p, k)` double loop. They accumulate contributions from all `(p, k)` pairs for a given `bin` candidate, and are reset to zero (or 1e-20 for `cosDen_d`) at the start of each `bin` iteration.

### Ordinal branch

Identical scope structure: `totalGain`, `cosNum_d`, `cosDen_d` are declared at `bin` scope (L1814, L1818, L1819), reset per `bin` candidate, accumulating over all `(p, k)` pairs.

### FindBestSplitPerPartition (Depthwise/Lossguide path)

In this function (L2240+), the structure differs:
- The outer loop is `feat` then `bin`
- Inside `bin`, the loop is `p` first, then `k`
- `gain`, `cosNum_d`, `cosDen_d` are declared at `(bin, p)` scope — **per partition** (L2279–2283, L2364–2368)
- This reflects the different semantic: each partition produces an independent best-split result
- The joint `continue` at L2293 (one-hot) and L2377 (ordinal) is the original unfixed form in BOTH sub-paths of `FindBestSplitPerPartition`

---

## 3. Parent-Term Locations

### Ordinal L2 (post S33-L4-FIX, L1954–1974)

```cpp
if (!wL_pos && !wR_pos) break;
if (wL_pos) totalGain += sL² / (wL + λ);
if (wR_pos) totalGain += sR² / (wR + λ);
totalGain -= totalSum² / (totalWeight + λ);   // L1973 — parent term
```

Parent term is subtracted **inside the `(p,k)` loop** (inside `for k`, inside `for p`), at `csv_train.cpp:1973`. It fires whenever at least one side is non-empty (`!wL_pos && !wR_pos` is the only escape). This means for the ordinal L2 path, the parent term is subtracted **once per `(p, k)` pair that contributes**.

### Ordinal Cosine (post S33-L4-FIX, L1976–2005)

```cpp
if (!wL_pos && !wR_pos) break;
double termNum = 0.0;
double termDen = 0.0;
if (wL_pos) { termNum += sL²*invL; termDen += sL²*wL*invL²; }
if (wR_pos) { termNum += sR²*invR; termDen += sR²*wR*invR²; }
cosNum_d += termNum;
cosDen_d += termDen;
```

**There is no parent-term subtraction** in the ordinal Cosine path. `cosNum_d` and `cosDen_d` accumulate only child-side contributions. The parent term concept does not apply to Cosine because the score is a ratio `cosNum_d / sqrt(cosDen_d)` rather than a differential. The comment at L1568–1569 states this explicitly:
> "No parent-node subtraction (unlike L2). Cosine is an absolute score, not a differential gain."

### One-hot L2 (current, L1700–1705)

```cpp
totalGain += (sL² / (wL + λ)) + (sR² / (wR + λ)) - (sP² / (wP + λ));
```

Parent term `sP² / (wP + λ)` is subtracted **inside the `(p,k)` loop** at L1704, same structure as ordinal L2. This is the **unfixed** one-hot branch — the joint `continue` at L1698 means the parent term is only subtracted when both sides are non-empty.

### One-hot Cosine (current, L1706–1719)

```cpp
cosNum_d += dSL*dSL*dInvL + dSR*dSR*dInvR;
cosDen_d += dSL*dSL*dWL*dInvL*dInvL + dSR*dSR*dWR*dInvR*dInvR;
```

**There is no parent-term subtraction** at L1706–1719, or anywhere in the one-hot Cosine path. This is consistent with the ordinal Cosine design: Cosine does not use a parent term. The one-hot Cosine path at L1706–1719 accumulates both `sL` and `sR` contributions into `cosNum_d`/`cosDen_d` in a single statement — there is no per-side masking here (the joint `continue` at L1698 either passes both sides or skips the entire `(p,k)` pair).

---

## 4. Joint Score Finalisation

### One-hot Cosine finalisation: L1726–1732

```cpp
// Finalize Cosine score after all partitions and dims are accumulated.
if (scoreFunction == EScoreFunction::Cosine) {
    totalGain = cosNum_d / std::sqrt(cosDen_d);   // L1731
}
```

This is the only finalisation. No additional parent-term subtraction. `cosNum_d / sqrt(cosDen_d)` is the entire score — it is computed once per `bin` candidate, after the full `(p, k)` double loop.

### Ordinal Cosine finalisation: L2050–2053

```cpp
if (scoreFunction == EScoreFunction::Cosine) {
    totalGain = cosNum_d / std::sqrt(cosDen_d);   // L2052
}
```

Same pattern. No additional parent-term subtraction.

### FindBestSplitPerPartition Cosine finalisation: L2320–2323 (one-hot), L2402–2405 (ordinal)

```cpp
if (scoreFunction == EScoreFunction::Cosine) {
    gain = cosNum_d / std::sqrt(cosDen_d);
}
```

Same pattern. No parent-term subtraction at finalization for any Cosine path.

---

## 5. `ComputeCosineGainKDim` Location and Live-Path Verdict

### Definition

`ComputeCosineGainKDim` is defined at `csv_train.cpp:1582–1587`:

```cpp
inline float ComputeCosineGainKDim(
    float totalNum,
    float totalDen
) {
    return totalNum / std::sqrt(totalDen);
}
```

It is a one-line thin wrapper: `totalNum / sqrt(totalDen)`. It was introduced in commit `83f30c3677` (S28-COSINE) as a helper to be called from `FindBestSplitPerPartition`.

### Live-path verdict

This function is **not the live production path** for any of the three `FindBestSplit` / `FindBestSplitPerPartition` branches as they exist now. The S28-OBLIV-DISPATCH refactor (`4083add248`) inlined the Cosine accumulation directly into each branch's `(p,k)` loop with local `cosNum_d`/`cosDen_d` double accumulators (S30-T2-KAHAN K4), and the finalisation is `cosNum_d / sqrt(cosDen_d)` inline at the end of the `bin` loop.

`ComputeCosineGainKDim` still exists in the file but is not called from any live code path. It is a dead helper (noted in DEC-033: "Dead scalar-signature helper `ComputeCosineGain` removed in `e0b0b1b527`"; `ComputeCosineGainKDim` itself is present but superseded by inline accumulation). Its signature takes pre-aggregated `float` numerator/denominator — incompatible with the current `double` per-term accumulation design.

The DEC-033 record at `DECISIONS.md:1499` confirmed this: "the live path at `S28-OBLIV-DISPATCH` may be inline" — and it is. S30/S31/S32/S33 work all targeted the inline accumulation loop, not `ComputeCosineGainKDim`.

There is no Metal kernel for Cosine gain scoring. The Cosine gain computation is entirely host-side CPU code in `csv_train.cpp`. The Metal kernel (`kernel_sources.h`) handles histogram accumulation only; `FindBestSplit` / `FindBestSplitPerPartition` run on CPU, reading the histogram output.

---

## 6. CPU Reference

### `TCosineScoreCalcer::AddLeafPlain` (`score_calcers.cpp:10–12`)

```cpp
void TCosineScoreCalcer::AddLeafPlain(int splitIdx,
    const TBucketStats& leftStats, const TBucketStats& rightStats) {
    NSimdOps::UpdateScoreBinKernelPlain(
        L2Regularizer, &rightStats.SumWeightedDelta,
        &leftStats.SumWeightedDelta, &Scores[splitIdx][0]);
}
```

This delegates entirely to `UpdateScoreBinKernelPlain`.

### `NGenericSimdOps::UpdateScoreBinKernelPlain` (`short_vector_ops.h:61–81`)

```cpp
inline void UpdateScoreBinKernelPlain(
    double scaledL2Regularizer,
    const NSimdOps::TValueType* trueStatsPtr,
    const NSimdOps::TValueType* falseStatsPtr,
    NSimdOps::TValueType* scoreBinPtr
) {
    const double trueAvrg  = CalcAverage(trueStatsPtr[0],  trueStatsPtr[1],  scaledL2Regularizer);
    const double falseAvrg = CalcAverage(falseStatsPtr[0], falseStatsPtr[1], scaledL2Regularizer);
    scoreBinPtr[0] += trueAvrg  * trueStatsPtr[0];
    scoreBinPtr[1] += trueAvrg  * trueAvrg  * trueStatsPtr[1];
    scoreBinPtr[0] += falseAvrg * falseStatsPtr[0];
    scoreBinPtr[1] += falseAvrg * falseAvrg * falseStatsPtr[1];
}
```

Where `CalcAverage(sumDelta, count, λ) = (count > 0) ? sumDelta / (count + λ) : 0` (`online_predictor.h:112–118`).

Expanding `trueAvrg = sumT / (wT + λ)`:
- `scoreBinPtr[0] += sumT² / (wT + λ)` — numerator contribution
- `scoreBinPtr[1] += sumT² * wT / (wT + λ)²` — denominator contribution

And same for `false` (left) side. Final score is `Scores[i][0] / sqrt(Scores[i][1])` (`score_calcers.h:58`).

**Parent-term subtraction: absent.** The CPU Cosine path accumulates both child sides unconditionally (with `CalcAverage` returning 0 for empty leaves, which is the per-side mask). No parent term is ever subtracted in `UpdateScoreBinKernelPlain` or in `AddLeaf`/`GetScores`. The `Scores` array is initialized to `{0, 1e-100}` and only receives positive additive contributions from child leaves.

**Per-partition call cadence**: `AddLeafPlain` (and hence `UpdateScoreBinKernelPlain`) is called once per leaf per approx dimension, from `CalcScoresForLeaf` (`leafwise_scoring.h:94`). For one-hot features, the CPU iterates over each bucket, computes `trueStats` = single-bucket stats and `falseStats` = allStats minus that bucket (`leafwise_scoring.h:126–133`), and calls `updateSplitScore(trueStats, falseStats, bucketIdx)`. This is per-(leaf, bin, dim) — there is no parent-node stats argument and no parent subtraction.

---

## 7. The Answer

**For the one-hot branch at `csv_train.cpp:1698`, is the parent term subtracted per-partition, per-(partition, k), or neither?**

**For Cosine: NEITHER.** There is no parent-term subtraction anywhere in the one-hot Cosine accumulation (L1706–1719) or its finalization (L1726–1731). The one-hot Cosine path accumulates `sL²*invL + sR²*invR` into `cosNum_d` and the corresponding denominator into `cosDen_d`, then computes `cosNum_d / sqrt(cosDen_d)`. This is consistent with the ordinal Cosine path (which also has no parent term) and with the CPU reference (`UpdateScoreBinKernelPlain` / `TCosineScoreCalcer`), which accumulates both child sides and returns a ratio — no parent subtraction anywhere.

**For L2: per-(partition, k).** The parent term `totalSum² / (totalWeight + λ)` is subtracted at L1704, which is inside both the `for p` loop (L1687) and the `for k` loop (L1688). It fires once per `(p, k)` pair whenever both sides are non-empty (the joint `continue` at L1698 is the gate).

**Citations:**
- One-hot Cosine accumulation (no parent term): `csv_train.cpp:1706–1719`
- One-hot L2 parent term subtraction (per-(p,k)): `csv_train.cpp:1704`
- Cosine finalization (no parent term added): `csv_train.cpp:1731`
- Comment confirming Cosine has no parent subtraction: `csv_train.cpp:1568–1569`
- CPU Cosine: no parent subtraction in `UpdateScoreBinKernelPlain`: `short_vector_ops.h:61–81`

---

## 8. Surprises and Inconsistencies

### Surprise 1: The S33-L4-FIX was applied to the ordinal branch but NOT to the one-hot branch

The S33-L4-FIX (per-side mask, Commit 1.5) landed in the ordinal `FindBestSplit` branch (L1941–2005). The one-hot `FindBestSplit` branch still carries the old joint `continue` at L1698. This is the asymmetry that T0b was asked to investigate.

**Implication for the load-bearing question**: Since the one-hot Cosine path has no parent term, the "mirror-applying the same patch" test that regressed the synthetic anchor by 3% cannot be explained by parent-term injection into `cosNum_d`. The regression must come from a different mechanism. The per-side fix for one-hot Cosine would change which `(p,k)` pairs contribute to `cosNum_d`/`cosDen_d` — specifically, it would add contributions from partitions where one side is empty (e.g., wR=0, wL=totalWeight) — but there is no parent-term term that would be injected or "doubled."

### Surprise 2: `FindBestSplitPerPartition` one-hot also retains the joint `continue`

In `FindBestSplitPerPartition` (Depthwise/Lossguide path, L2272–2338), the one-hot Cosine branch at L2300–2313 also uses the old joint pattern:

```cpp
if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;  // L2293
cosNum_d += dSL*dSL*dInvL + dSR*dSR*dInvR;
cosDen_d += ...
```

The ordinal `FindBestSplitPerPartition` path (L2377) also retains the joint `continue`. Neither sub-path in `FindBestSplitPerPartition` received the S33-L4-FIX per-side mask. The S33-L4-FIX was scoped only to `FindBestSplit` ordinal (the SymmetricTree path).

### Surprise 3: The joint `continue` in one-hot L2 subtracts the parent term inside the skip guard

At L1698–1704: the joint `continue` gates the entire `switch` block including the L2 parent-term subtraction. This means:
- If `wR == 0`: the parent term `sP²/(wP+λ)` is NOT subtracted for this `(p,k)` pair
- For the ordinal L2 post-S33-FIX (L1966–1973): the parent term IS subtracted even when one side is empty (only escaped when `!wL_pos && !wR_pos`)

The one-hot L2 path has the same asymmetry as ordinal L2 pre-fix. The one-hot L2 at L1700–1705 and the Cosine at L1706–1719 are both behind the same `continue` at L1698.

### Surprise 4: `ComputeCosineGainKDim` is a dead helper

The function at L1582–1587 is still compiled but not called by any live path. It was written as a callable helper in S28 but was rendered dead by the S30 K4 refactor that inlined double-precision accumulation directly into each branch. This is mildly dangerous if future refactors reintroduce calls to it — it would silently downcast to float. It should be removed or marked with a `[[deprecated]]` annotation.

### Surprise 5: The Cosine formula comment at L1554–1578 says "No parent-node subtraction" explicitly

The inline comment block documenting the Cosine gain formula (`csv_train.cpp:1568–1569`) states:
> "No parent-node subtraction (unlike L2). Cosine is an absolute score, not a differential gain."

This is correct and consistent with the CPU reference. @mathematician's T0a framing asks whether a parent term could be injected by the fix — the code says no: the Cosine formula structurally has no parent term, and the `continue`/per-side-mask distinction only affects which child-side contributions are included, not whether a parent term exists.
