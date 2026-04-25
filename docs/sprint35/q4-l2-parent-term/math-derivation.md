# S35-Q4 T0 — L2 parent-term divergence between MLX and CPU

**Mode:** Pragmatic (applied; ground truth = the actual code)
**Scope:** `FindBestSplit` L2 path, one-hot branch (L1654–1770) and ordinal branch (L1772–2099) of `catboost/mlx/tests/csv_train.cpp` vs `TL2ScoreCalcer::AddLeafPlain/Ordered` in `catboost/private/libs/algo/score_calcers.cpp`.
**Author:** mathematician
**Date:** 2026-04-25
**Ticket:** #128 (S35-Q4-L2-PARENT-TERM), deferred from S34-PROBE-F-LITE T0a §8.2 Q4

> **TL;DR.** MLX's per-(p, k) parent-term subtraction `−totalSum²/(totalWeight+λ)` at `csv_train.cpp:1973` (ordinal L2) and `:1704` (one-hot L2) is a **constant offset across all (feat, bin) candidates within a single `FindBestSplit` invocation** for the **ordinal path** (post S33-L4-FIX), and is therefore **argmax-invariant** vs CPU's no-parent-subtraction `TL2ScoreCalcer`. The constant is `c_step = −Σ_p Σ_k Sum_{p,k}²/(W_{p,k}+λ)`, depending only on partition totals `(Sum_{p,k}, W_{p,k}) = perDimPartStats[k][p]` which are **identically constant across (feat, bin)**. The "both-sides-empty" gating that could in principle break this constancy fires **only on fully-empty partitions** (`totalWeight ≤ 2·1e-15`), where CPU's masked `CalcAverage` also contributes 0 — preserving the constant offset.
>
> The **one-hot L2 path** (csv_train.cpp:1698, joint-skip not yet replaced by S33-L4-FIX) does **not** enjoy this property: when `wL=0` xor `wR=0` (a normal occurrence for one-hot bins), MLX joint-skips both the leaf-add and the parent-sub, while CPU adds the non-empty side's contribution. Δ(feat, bin) becomes bin-dependent. However, the dominant divergence here is the **joint-skip** pathology (the same class as DEC-042 for Cosine, ordinal L2), not the parent term per se. After the joint-skip is replaced with a per-side mask, the parent-term subtraction collapses to a constant offset and becomes argmax-invariant for one-hot L2 too.
>
> **Verdict:** **Argmax-invariant in all practical configurations** for the ordinal L2 path (post S33-L4-FIX). Confidence: **high**. For the one-hot L2 path, the parent term is currently mixed with a separate joint-skip bug class; once the joint-skip is replaced (one-hot L2 sibling of S33-L4-FIX Commit 1.5), the parent-term offset collapses to constant. Confidence: **high**, contingent on the one-hot L2 joint-skip being fixed.
>
> **Recommendation:** Close S35-Q4 on math (no T1 probe needed) for the ordinal path. Open a side ticket to extend S33-L4-FIX Commit 1.5 (per-side mask) to the one-hot L2 branch — that is a separate DEC-042-class fix and not strictly a "parent term" fix.

---

## 1. Setup — code citations

### 1.1 MLX L2 contribution per (feat, bin) candidate

#### Ordinal branch (post S33-L4-FIX), `csv_train.cpp:1812–2107`

The `for p × for k` double loop sits inside the per-bin `totalGain` accumulator (initialised L1814 to `0.0`, read for argmax at L2106-class assignment). For L2 the case body at L1954-1975 is:

```cpp
case EScoreFunction::L2: {
    if (!wL_pos && !wR_pos) break;                                  // L1966 — both-sides-empty gate
    if (wL_pos) {
        totalGain += (sumLeft  * sumLeft ) / (weightLeft  + l2RegLambda);   // L1968
    }
    if (wR_pos) {
        totalGain += (sumRight * sumRight) / (weightRight + l2RegLambda);   // L1971
    }
    totalGain -= (totalSum * totalSum) / (totalWeight + l2RegLambda);       // L1973  ← PARENT TERM
    break;
}
```

with `wL_pos := (weightLeft > 1e-15f)` (L1950) and `wR_pos := (weightRight > 1e-15f)` (L1951). `totalSum := perDimPartStats[k][p].Sum` and `totalWeight := perDimPartStats[k][p].Weight` (L1879-1880) are the **partition totals** for partition p, dimension k — they do **not** depend on `featIdx` or `bin`.

#### One-hot branch (still joint-skip, pre S33-L4-FIX-for-1H-L2), `csv_train.cpp:1654–1770`

```cpp
if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;          // L1698 — JOINT SKIP
switch (scoreFunction) {
    case EScoreFunction::L2:
        totalGain += (sumLeft  * sumLeft ) / (weightLeft  + l2RegLambda)
                   + (sumRight * sumRight) / (weightRight + l2RegLambda)
                   - (totalSum * totalSum) / (totalWeight + l2RegLambda);   // L1702-1704
        break;
```

Same parent-term formula; the gating is the older joint-skip — distinct from the per-side mask shipped in the ordinal branch. No `wL_pos`/`wR_pos` split here yet.

### 1.2 CPU L2 contribution per leaf, `score_calcers.cpp:20-49`

```cpp
void TL2ScoreCalcer::AddLeafPlain(int splitIdx, const TBucketStats& leftStats, const TBucketStats& rightStats) {
    const double rightAvrg = CalcAverage(rightStats.SumWeightedDelta, rightStats.SumWeight,  L2Regularizer);  // L21-25
    const double leftAvrg  = CalcAverage(leftStats.SumWeightedDelta,  leftStats.SumWeight,   L2Regularizer);  // L26-30
    AddLeaf(splitIdx, rightAvrg, rightStats);                                                                  // L31
    AddLeaf(splitIdx, leftAvrg,  leftStats);                                                                   // L32
}
```

Where `AddLeaf` (`score_calcers.h:87-89`) does:

```cpp
void AddLeaf(int splitIdx, double leafApprox, const TBucketStats& leafStats) override {
    Scores[splitIdx] += leafApprox * leafStats.SumWeightedDelta;
}
```

And `CalcAverage` (verified by inspection of the SSE-equivalent `UpdateScoreBinKernelPlain` at `short_vector_ops.h:155–175` line `isSumWeightPositive = _mm_cmpgt_pd(sumWeight, _mm_setzero_pd())` masking the divide) returns:

```
CalcAverage(sumDelta, sumWeight, λ) = (sumWeight > 0) ? sumDelta / (sumWeight + λ) : 0
```

So per-leaf CPU contribution to `Scores[splitIdx]` is:

```
contrib(leaf, side) = (sumWeight > 0) · sumDelta² / (sumWeight + λ)        [side ∈ {L, R}]
```

with `Scores[splitIdx]` accumulated over all `2^d` leaves of the current oblivious tree (one `AddLeafPlain` call per leaf, see `scoring.cpp:573-584` `UpdateSplitScore` inside `UpdateScores` at L673-683, which iterates `for leaf in 0..leafCount`).

**Crucially: there is no parent-term subtraction anywhere in `TL2ScoreCalcer::AddLeafPlain`, `AddLeaf`, the SSE2 kernel, or the reducer.** The CPU L2 score is the **absolute** sum of squared-leaf-averages-weighted-by-sum, not a differential gain. The argmax picker (`tensor_search_helpers.cpp:716-789`, `SetBestScore`) operates directly on `GetScores()` values without any further normalisation.

### 1.3 Index correspondence MLX ↔ CPU

| Concept                       | MLX                                | CPU                                                   |
|-------------------------------|------------------------------------|-------------------------------------------------------|
| candidate (feat, bin)         | `(featIdx, bin)`                   | `(candidate, splitIdx)`                               |
| leaf of current tree          | partition `p ∈ 0..numPartitions`   | `leaf ∈ 0..leafCount` (`leafCount = 2^depth`)         |
| approx dimension              | `k ∈ 0..K`                         | `dim ∈ 0..approxDimension` (looped at `scoring.cpp:751`)|
| partition totals              | `perDimPartStats[k][p].{Sum,Weight}` | `(falseStats+trueStats).SumWeightedDelta / SumWeight` |
| left/right (for split)        | `sumLeft, sumRight, weightLeft, weightRight` | `leftStats, rightStats`                       |

The MLX `(p, k)` double loop and the CPU `(leaf, dim)` double loop accumulate over the same set; the L2 `Scores[splitIdx]` is one scalar per candidate, summed over both indices.

---

## 2. MLX L2 contribution — explicit formula

For the **ordinal branch** (post S33-L4-FIX), the MLX score for candidate `(f, b)`:

```
S^MLX(f, b) =  Σ_p Σ_k  [ 𝟙{wL_pos} · sL²_{f,b,p,k}/(wL_{f,b,p,k}+λ)
                         + 𝟙{wR_pos} · sR²_{f,b,p,k}/(wR_{f,b,p,k}+λ)
                         − 𝟙{wL_pos ∨ wR_pos} · Sum²_{p,k}/(W_{p,k}+λ) ]
```

where:
- `Sum_{p,k} := perDimPartStats[k][p].Sum`, `W_{p,k} := perDimPartStats[k][p].Weight` — **independent of (f, b)**.
- `sL_{f,b,p,k}, sR_{f,b,p,k}, wL_{f,b,p,k}, wR_{f,b,p,k}` — depend on (f, b) via `histData[bin]` (one-hot) or `suffGrad[base+bin]` (ordinal, L1883).
- `wL_pos := wL_{f,b,p,k} > 1e-15f`, similarly `wR_pos`.
- The indicator `𝟙{wL_pos ∨ wR_pos}` reflects the L1966 `if (!wL_pos && !wR_pos) break;` early-out: when both sides are empty, the entire case body skips, including the parent subtraction.

For the **one-hot branch** (still joint-skip):

```
S^MLX_1H(f, b) =  Σ_p Σ_k  𝟙{wL_pos ∧ wR_pos} · [ sL²_{f,b,p,k}/(wL_{f,b,p,k}+λ)
                                                  + sR²_{f,b,p,k}/(wR_{f,b,p,k}+λ)
                                                  − Sum²_{p,k}/(W_{p,k}+λ) ]
```

---

## 3. CPU L2 contribution — explicit formula

For both branches (CPU does not distinguish one-hot vs ordinal at the score-calcer level — `AddLeafPlain` is invoked per leaf with `(leftStats, rightStats)` populated by `CalcScoresForLeaf` from the bucket stats):

```
S^CPU(f, b) =  Σ_leaf Σ_dim  [ 𝟙{wL_{f,b,leaf,dim} > 0} · sL²_{f,b,leaf,dim}/(wL_{f,b,leaf,dim}+λ)
                              + 𝟙{wR_{f,b,leaf,dim} > 0} · sR²_{f,b,leaf,dim}/(wR_{f,b,leaf,dim}+λ) ]
```

Note: CPU's mask is `> 0` (strict), MLX's mask is `> 1e-15f`. For practical purposes these agree up to a numerical zero; the ε-band `[0, 1e-15]` is empirically empty in fp32 accumulators. Treated as identical for this analysis.

**No parent-term subtraction.** Verified by exhaustive grep of `score_calcers.cpp` and `score_calcers.h`: no `totalSum`, no `Sum²/(W+λ)`, no `parentSum`, no `baseGain` — only `Scores[splitIdx] += leafApprox * SumWeightedDelta` accumulation. The reducer in `scoring.cpp:1013-1050` (`GetScores`) and the picker in `tensor_search_helpers.cpp:716-789` (`SetBestScore`) likewise contain no parent-term logic.

---

## 4. depth = 0, numPartitions = 1 case — algebraic proof of constant offset

At depth 0 there is exactly one partition (the root), so `numPartitions = 1`. Let `K = 1` for clarity (the multi-K case is identical with an extra Σ_k over a fixed K-set).

For any candidate `(f, b)`:
- CPU: `S^CPU(f, b) = sL²/(wL+λ) + sR²/(wR+λ)` (with masking when one side is 0).
- MLX (ordinal, post-fix): `S^MLX(f, b) = sL²/(wL+λ) + sR²/(wR+λ) − Sum²/(W+λ)` (with the same per-side masking and the both-sides-empty early-out).

The parent term `Sum²/(W+λ)` uses **only** root-partition totals — these are the **dataset-level statistics for the current step**: `Sum = Σ_{i in dataset} weighted_delta_i`, `W = Σ_{i in dataset} weight_i`. They depend on the iteration (the model's current residuals) but **not** on `(f, b)`. Therefore:

```
Δ(f, b) = S^MLX(f, b) − S^CPU(f, b) = − Sum² / (W + λ)  =  c_step
```

a constant across (f, b).

The argmax over (f, b) is invariant under additive constants:

```
argmax_{f,b} S^MLX(f, b)  =  argmax_{f,b} (S^CPU(f, b) + c_step)  =  argmax_{f,b} S^CPU(f, b)
```

**Argmax-invariant.** ∎

The both-sides-empty gate `if (!wL_pos && !wR_pos) break;` cannot fire at depth=0 with a non-trivial dataset: it requires `wL ≤ 1e-15 ∧ wR ≤ 1e-15 ⇒ W = wL + wR ≤ 2·1e-15`, i.e. an essentially empty dataset. Any standard training run has `W > 0` at the root, so the gate is dormant. CPU's `AddLeafPlain` on an essentially empty dataset returns zero contributions for both sides too. No edge case.

---

## 5. depth ≥ 1 OR numPartitions > 1 — does the parent term vary per (feat, bin)?

This is the load-bearing case. We separate by branch.

### 5.1 Ordinal branch (post S33-L4-FIX)

Define the set of `(p, k)` pairs reached by the parent-term subtraction for candidate `(f, b)`:

```
A(f, b) := { (p, k) : wL_pos(f, b, p, k) ∨ wR_pos(f, b, p, k) }
```

i.e. the (p, k) pairs where the L1966 early-out does **not** fire. The MLX-vs-CPU offset is then:

```
Δ(f, b) =  − Σ_{(p,k) ∈ A(f,b)}  Sum_{p,k}² / (W_{p,k} + λ)
```

(The leaf-add part of MLX and CPU agree term-by-term on `A(f, b)`, and both contribute zero outside `A(f, b)`: outside `A(f, b)` we have `wL ≤ 1e-15 ∧ wR ≤ 1e-15`, hence both indicators in CPU's formula are zero.)

**Claim:** `A(f, b)` is the same set across all `(f, b)`, namely

```
A(f, b)  =  A_step  :=  { (p, k) : W_{p,k} > 1e-15 · 2 }  =  { (p, k) : partition p is non-empty }
```

**Proof.** Fix `(p, k)`. Note `W_{p,k} = wL_{f,b,p,k} + wR_{f,b,p,k}` (since `weightLeft = totalWeight − weightRight` at L1886). Two cases:

(i) `W_{p,k} > 2·1e-15`: then at least one of `wL, wR` exceeds `1e-15` (pigeonhole on a sum > 2·1e-15 split into two non-negatives), regardless of how the weight splits between L/R. Hence `wL_pos ∨ wR_pos` holds, i.e. `(p, k) ∈ A(f, b)` for every `(f, b)`.

(ii) `W_{p,k} ≤ 2·1e-15`: then both `wL ≤ W ≤ 2·1e-15` and `wR ≤ W ≤ 2·1e-15`. With the threshold `1e-15f`, both `wL_pos` and `wR_pos` are at the boundary; in practice (fp32 with standard residual magnitudes), both fail. Hence `(p, k) ∉ A(f, b)` for every `(f, b)`.

In both cases the membership of `(p, k)` in `A(f, b)` is determined entirely by `W_{p,k}`, which depends only on `(p, k)`, not on `(f, b)`. Therefore `A(f, b) = A_step`, a constant set. ∎

(The thin `1e-15 < W ≤ 2·1e-15` corner where `wL` and `wR` could each individually be below threshold despite their sum exceeding it is a fp32-noise band of essentially zero measure; under any non-degenerate training distribution it is empirically empty. Flagged in §9 as a residual code-reading dependency, but not a real divergence source.)

Substituting:

```
Δ(f, b) = −Σ_{(p,k) ∈ A_step} Sum_{p,k}² / (W_{p,k} + λ)  =  c_step
```

**a constant in (f, b).** Argmax is preserved.

### 5.2 One-hot branch (joint-skip, NOT yet S33-L4-FIX'd)

Define instead:

```
B(f, b) := { (p, k) : wL_pos(f, b, p, k) ∧ wR_pos(f, b, p, k) }
```

— the set of (p, k) pairs that pass the joint-skip at L1698. For candidate `(f, b)`:

```
S^MLX_1H(f, b) =  Σ_{(p,k) ∈ B(f,b)}  [ sL²/(wL+λ) + sR²/(wR+λ) − Sum²_{p,k}/(W_{p,k}+λ) ]
S^CPU(f, b)    =  Σ_{(p,k) ∈ A_step}  [ 𝟙{wL>0} sL²/(wL+λ) + 𝟙{wR>0} sR²/(wR+λ) ]
              =  Σ_{(p,k) : wL>0} sL²/(wL+λ) + Σ_{(p,k) : wR>0} sR²/(wR+λ)
```

The set `B(f, b)` is **(f, b)-dependent**. For one-hot category bin `b`:

- `wR_{f,b,p,k} = histData[firstFold + b]` (L1693) = weight of category `b` in partition `p`, dim `k`.
- `wL_{f,b,p,k} = W_{p,k} − wR_{f,b,p,k}`.
- `wR_pos` ⇔ at least one doc in partition `p` has feature `f` taking category `b`.
- `wL_pos` ⇔ at least one doc in partition `p` has feature `f` ≠ category `b`.

Both depend on `(f, b)`. Therefore `B(f, b)` varies. Concretely, splitting Δ into "leaf" and "parent" terms:

```
Δ_1H(f, b)  =  S^MLX_1H(f, b) − S^CPU(f, b)
            = − [ Σ_{(p,k) ∈ A_step \ B(f,b), wR>0} sR²/(wR+λ)            ]   ← leaf-add MLX dropped
              − [ Σ_{(p,k) ∈ A_step \ B(f,b), wL>0} sL²/(wL+λ)            ]   ← leaf-add MLX dropped
              − [ Σ_{(p,k) ∈ B(f,b)} Sum²_{p,k}/(W_{p,k}+λ)                ]   ← parent-sub MLX added
```

The first two lines are the **joint-skip pathology** (dropping non-empty side contributions when the other side is empty) — same class as DEC-042 for Cosine, ordinal L2. The third line is the parent-term offset, but its **support** `B(f, b)` is now `(f, b)`-dependent: when partition `p` has all docs in category `b` (so `wL = 0` at bin `b`), `(p, k) ∉ B(f, b)` and parent isn't subtracted; for a different `b'` with `wL > 0 ∧ wR > 0` it is.

So Δ_1H(f, b) **does** vary across (f, b). It is **bin-dependent** (different bins of the same feature pick out different categories, hence different empty-side patterns) and **feat-dependent** (different categorical features have different category distributions, hence different empty-side patterns).

**However**, the dominant divergence is the joint-skip pathology, not the parent term. Once the per-side mask is applied (one-hot L2 sibling of S33-L4-FIX Commit 1.5 — straightforward edit at L1698 mirroring L1950–1973), the formula collapses to:

```
S^MLX_1H_fixed(f, b) =  Σ_{(p,k) ∈ A_step}  [ 𝟙{wL_pos} sL²/(wL+λ) + 𝟙{wR_pos} sR²/(wR+λ) − Sum²_{p,k}/(W_{p,k}+λ) ]
```

— identical structure to the ordinal branch. Δ collapses to `c_step` as in §5.1, and argmax-invariance is restored.

### 5.3 Across-step (between-iteration) behaviour

Between training iterations, `Sum_{p,k}` and `W_{p,k}` change as residuals update and the tree structure evolves. So `c_step` is iteration-dependent, but is **the same scalar across all candidates evaluated within one `FindBestSplit` call**. Argmax invariance is established within a step; cross-step magnitudes (e.g., for early-stopping thresholds based on `bestSplit.Score`) would see a shifted score but no argmax change.

---

## 6. Verdict

| Path                                        | Δ(f, b) structure                                                                    | Argmax behaviour                                  | Confidence |
|---------------------------------------------|--------------------------------------------------------------------------------------|---------------------------------------------------|------------|
| Ordinal L2 (post S33-L4-FIX, csv_train.cpp:1973) | `Δ = c_step = −Σ_{(p,k) ∈ A_step} Sum²_{p,k}/(W_{p,k}+λ)`, **constant in (f, b)** | **Argmax-invariant** vs CPU                       | High       |
| Depth=0 / numPartitions=1, ordinal L2       | `Δ = −Sum² / (W + λ)`, single-term constant                                          | **Argmax-invariant**                              | High       |
| One-hot L2 (joint-skip, csv_train.cpp:1704) | `Δ_1H(f, b)` varies; mixes joint-skip leaf-drop AND parent-term over a (f, b)-dep set `B(f, b)` | **Argmax-biased**, dominantly by joint-skip       | High (that it varies); the parent term *per se* is the second-order issue |
| One-hot L2 (after applying per-side mask)   | `Δ = c_step` (same as ordinal)                                                       | **Argmax-invariant**                              | High (subject to applying the fix)                |

**Summary:** The MLX L2 parent-term subtraction is **option (1)** from the task brief: equivalent to CPU's no-subtraction L2 *up to a constant offset*, hence argmax-invariant — for the ordinal path. For the one-hot path, the apparent parent-term divergence is actually entangled with a separate joint-skip bug class (DEC-042 sibling); resolve the joint-skip and the parent term reduces to a constant offset.

---

## 7. Predicted empirical signature

If the parent term were argmax-biasing (option 2 from the task brief, which our analysis rules out for the ordinal path), we would expect:

- **Configuration that would surface bias:** `depth ≥ 2` (so `numPartitions ≥ 4`) on a dataset where partition weights `W_{p,k}` differ markedly between leaves of the current tree (i.e. heavily unbalanced splits in the prior iteration) — combined with a feature whose split candidates have similar raw gains (so a small offset could flip argmax). The 18-config G4d gate uses depth=6 and small datasets (50k/100k rows), so depth-coverage is fine; what G4d *misses* is **split-selection agreement**, since it measures aggregate RMSE drift after `n_iters` iterations rather than per-iteration bestSplit (featIdx, binId) equality.
- **What the math predicts instead:** Because Δ(f, b) is constant across (f, b), the bestSplit.featIdx and bestSplit.binId chosen by MLX must equal CPU's for every iteration of every config, *modulo the separate joint-skip bug on the one-hot L2 path*. The 18-config G4d aggregate-RMSE gate therefore correctly sees no parent-term contribution to drift on the ordinal L2 path.
- **Empirical test that would falsify this analysis:** Run a depth=6, 50k/RMSE/128b synthetic config with **ordinal-only features** and `score_function=L2`. Compare per-iteration `bestSplit.{featIdx, binId}` between MLX and CPU. Prediction: **bit-exact match every iteration** (modulo fp32 tie-breaking on near-equal candidates, which has its own jitter unrelated to the parent term). If a mismatch appears whose `(f, b)` correlates with `Sum²_{p,k}/(W_{p,k}+λ)` magnitude — which it cannot under this analysis — the proof is wrong.

The 18-config L2 parity gate G4d (per-iteration aggregate RMSE drift) does **not** test this property directly: it sees Δ(f, b) propagate through tree structure into final predictions, but a constant Δ(f, b) across (f, b) yields **identical tree structures** to CPU, which then produces **identical predictions**, which then produces **zero RMSE drift**. So G4d is consistent with the math: no drift attributable to the parent term for ordinal L2.

For the **one-hot L2 path**, G4d has a real blind spot: it would catch *aggregate* RMSE drift but not localise it to "joint-skip vs parent term". The empirical signature of joint-skip dominance vs parent-term dominance differs structurally (joint-skip drops leaf-add terms only on bins with one-side-empty partitions; parent term offsets only those bins' parent-sub) and could be discriminated by a per-bin probe.

---

## 8. Recommended fix shape

**For the ordinal L2 path (csv_train.cpp:1973):** **No fix needed.** The parent-term subtraction is a constant offset across (f, b), preserves argmax, and matches CPU's L2 score *up to that offset*. Document this property in a comment near L1973 to prevent future confusion (e.g. "`-totalSum²/(totalWeight+λ)` is a step-constant offset in (f, b); CPU L2 omits it; argmax invariant").

If a code-aesthetic alignment with CPU is desired (so MLX's `bestSplit.Gain`/`Score` magnitudes match CPU's `Scores[splitIdx]` numerically rather than only up to offset), one could:

1. **Remove the parent subtraction at L1973** entirely. Then MLX's S^MLX(f, b) = S^CPU(f, b) bit-up-to-fp32-roundoff. This is the simplest change. Cost: changes the *absolute* gain magnitudes recorded in `bestSplit.Gain` and `bestSplit.Score`, which may be load-bearing for downstream code (e.g., model serialisation, logging, early stopping). Audit needed.

2. **Move the parent subtraction to a finalisation step at bin scope** (after the `(p, k)` loop closes, line ~2045). The parent term is `Σ_{(p,k) ∈ A_step} Sum²_{p,k}/(W_{p,k}+λ)`, computable once per `FindBestSplit` invocation and subtracted as a constant. This makes the constancy explicit and avoids `numPartitions × K` redundant subtractions per bin. Performance gain: marginal (one fp64 op per (p, k) per bin → one per bin). Cleanliness gain: high. Cost: one extra accumulator at function scope.

3. **No-op (status quo).** Argmax is correct, downstream consumers of `bestSplit.Gain` already handle the offset (or don't care). If gain magnitudes are not load-bearing externally, this is the lowest-risk option.

**Recommended:** Option 3 (no-op) for the ordinal L2 path, with a clarifying comment. The L2 parity gate G4d already passes; the math says it should; we close S35-Q4 here.

**For the one-hot L2 path (csv_train.cpp:1698, 1702-1704):** **Apply S33-L4-FIX Commit 1.5 sibling fix.** Replace the joint-skip `if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;` at L1698 with the per-side mask pattern from L1966-1973:

```cpp
const bool wL_pos = (weightLeft  > 1e-15f);
const bool wR_pos = (weightRight > 1e-15f);
switch (scoreFunction) {
    case EScoreFunction::L2: {
        if (!wL_pos && !wR_pos) break;
        if (wL_pos) totalGain += (sumLeft  * sumLeft ) / (weightLeft  + l2RegLambda);
        if (wR_pos) totalGain += (sumRight * sumRight) / (weightRight + l2RegLambda);
        totalGain -= (totalSum * totalSum) / (totalWeight + l2RegLambda);
        break;
    }
    case EScoreFunction::Cosine: { /* keep current joint-skip — see S34 verdict */ }
    ...
}
```

This is **separate from the parent-term question** — it is the same DEC-042 fix class. After this fix, the one-hot L2 parent-term offset collapses to `c_step`, identical to the ordinal branch.

**Caution:** The S34-PROBE-F-LITE verdict established that the per-side mask is **wrong** for one-hot **Cosine** (it adds spurious `Sum²/(W+λ)` to `cosNum` and `cosDen` because Cosine has *no* parent term to cancel). For one-hot **L2**, the parent term IS subtracted, so the per-side mask DOES cancel correctly — making the L2 fix safe where the Cosine fix is harmful. Apply per-side mask **only to the one-hot L2 case**, not Cosine. The ordinal branch already has this asymmetry handled correctly.

This one-hot L2 fix is its own ticket (call it `#129 S35-Q4-1H-L2-PER-SIDE` if scoped) — not strictly under the "parent term" Q4 banner but a discovered sibling.

---

## 9. Open questions

1. **Cross-step magnitude consumers of `bestSplit.Gain` / `bestSplit.Score`.** Need to confirm whether any downstream code reads the absolute gain magnitude (rather than the argmax). Candidates: early-stopping logic, model serialisation that records per-tree gains, debug logging. If any such consumer exists and assumes CPU-equivalent magnitudes, fix option (1) or (2) becomes load-bearing, and option (3) silently misreports. **Action:** grep for `bestSplit.Gain`, `bestSplit.Score`, `BestScore`, `BestGain` in MLX and CPU paths to confirm. Out of scope for T0.

2. **fp32 boundary `1e-15 < W ≤ 2·1e-15` corner.** §5.1 noted that the strict claim `A(f, b) = A_step` rests on this thin band being empirically empty. For pathological datasets with extreme weight concentration (e.g., zero-weighted samples being given exactly `5e-16` weight), the band could in principle be non-empty. **Action:** if S35 ever runs on zero-weight or near-zero-weight datasets (auxiliary tasks, rare sample reweighting), revisit. Practical impact: nil for current configs.

3. **Multi-target / `K > 1` interaction with parent term.** The Σ_k loop is benign because `Sum_{p,k}` is per-(p, k); the proof in §5.1 quantifies over (p, k) jointly. No K-specific corner expected. **Action:** none, but flagged for completeness.

4. **One-hot L2 per-side-mask sibling fix scope.** Recommended in §8 but unproven in this T0. The math here only proves "if the joint-skip is replaced, the parent-term reduces to a constant offset". The actual fix correctness (no Cosine-style spurious-injection) follows from the parent-term subtraction's algebraic role: in L2 it cancels exactly the `Sum²/(W+λ)` injected by the per-side mask on otherwise-skipped (p, k). **Action:** if pursued, write a 1-page sibling derivation proving that the per-side mask + parent-term combination on one-hot L2 yields the same `S^CPU(f, b)` as CPU + constant offset, mirroring §5.1 for the one-hot index structure. Should be ~10 lines of algebra (no new ideas).

---

## 10. References

- `catboost/mlx/tests/csv_train.cpp:1654-1770` (one-hot branch, L2 at 1702-1704; joint-skip at 1698)
- `catboost/mlx/tests/csv_train.cpp:1812-2107` (ordinal branch, L2 at 1954-1975 incl. parent-term L1973; per-side mask shipped S33-L4-FIX Commit 1.5)
- `catboost/private/libs/algo/score_calcers.cpp:20-49` (CPU `TL2ScoreCalcer::AddLeafPlain/Ordered`)
- `catboost/private/libs/algo/score_calcers.h:76-97` (CPU `TL2ScoreCalcer::AddLeaf`, `GetScores`)
- `catboost/libs/helpers/short_vector_ops.h:155-175` (CPU SSE2 `UpdateScoreBinKernelPlain`, used by Cosine — confirms `isSumWeightPositive` mask pattern; L2 calls `AddLeaf` directly, not this kernel)
- `catboost/private/libs/algo/scoring.cpp:573-723` (CPU `UpdateScores`, leaf loop at 673-683)
- `catboost/private/libs/algo/scoring.cpp:1013-1050` (CPU `GetScores`, scaledL2Regularizer formula at 1031)
- `catboost/private/libs/algo/tensor_search_helpers.cpp:716-789` (CPU `SetBestScore` argmax — confirms no parent-term in picker)
- `catboost/private/libs/algo/greedy_tensor_search.cpp:678-737` (CPU candidate loop — one `MakePointwiseScoreCalcer` per candidate, no parent-term injection at this level)
- `docs/sprint34/probe-f-lite/math-derivation.md` §8.2 Q4 (deferred ticket origin)
- DEC-042 (per-side mask for ordinal Cosine and ordinal L2); DEC-036 (root cause of cumulative drift)
