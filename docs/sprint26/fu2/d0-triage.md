# S26-FU-2 T1 Triage: CPU RandomStrength Mechanism

**Date**: 2026-04-22
**Branch**: `mlx/sprint-26-fu2-noise-dwlg`
**Source audited**: `catboost/private/libs/algo/greedy_tensor_search.cpp`

---

## Point 1: CalcDerivativesStDevFromZeroPlainBoosting accepts no partition/leaf subset

```cpp
// greedy_tensor_search.cpp:92–107
static double CalcDerivativesStDevFromZeroPlainBoosting(
    const TFold& fold,
    NPar::ILocalExecutor* localExecutor
) {
    Y_ASSERT(fold.BodyTailArr.size() == 1);
    Y_ASSERT(fold.BodyTailArr.front().WeightedDerivatives.size() > 0);
    const auto& weightedDerivatives = fold.BodyTailArr.front().WeightedDerivatives;
    double sum2 = 0;
    for (const auto& perDimensionWeightedDerivatives : weightedDerivatives) {
        sum2 += L2NormSquared<double>(perDimensionWeightedDerivatives, localExecutor);
    }
    return sqrt(sum2 / weightedDerivatives.front().size());
}
```

Confirmed: takes the entire fold (all docs in the single BodyTailArr), no
partition or leaf subset parameter. Returns one `double` scalar.

---

## Point 2: CalcScoreStDev is called exactly once per tree, before the depth loop, at lines :1186, :1480, :1776

**Oblivious (:1186)** — `GreedyTensorSearchOblivious`, before `for (ui32 curDepth = ...)`:
```cpp
// greedy_tensor_search.cpp:1186
const double scoreStDev = CalcScoreStDev(learnSampleCount, modelLength, *fold, ctx);
// ...
for (ui32 curDepth = 0; curDepth < ctx->Params.ObliviousTreeOptions->MaxDepth; ++curDepth) {
    // scoreStDev passed unchanged into CalcScores(... scoreStDev ...)
```

**Depthwise (:1480)** — `GreedyTensorSearchDepthwise`, before `for (ui32 curDepth = ...)`:
```cpp
// greedy_tensor_search.cpp:1480
const double scoreStDev = CalcScoreStDev(learnSampleCount, modelLength, *fold, ctx);
// depth loop follows; scoreStDev is captured into TSubtractTrickInfo at :1511
```

**Lossguide (:1776)** — `GreedyTensorSearchLossguide`, before priority-queue loop:
```cpp
// greedy_tensor_search.cpp:1776
const double scoreStDev = CalcScoreStDev(learnSampleCount, modelLength, *fold, ctx);
// passed into TSubtractTrickInfo at :1800 and into FindBestCandidate at :1892
```

All three: one call per tree, scalar declared `const`, never recomputed.

---

## Point 3: scoreStDev flows unchanged into per-partition / per-leaf scoring

- **Oblivious**: `CalcScores(data, currentSplitTree, scoreStDev, ...)` at :1199 — same
  scalar every depth, every partition.
- **Depthwise**: `TSubtractTrickInfo subTrickInfo(..., scoreStDev, ...)` at :1511 — passed
  as `double`, no recomputation per leaf.
- **Lossguide**: `TSubtractTrickInfo subTrickInfo(..., scoreStDev, ...)` at :1800 and
  `FindBestCandidate(..., scoreStDev, ...)` at :1892 — same scalar from tree start through
  all leaves in the priority queue.

No partition-local rescaling or per-leaf re-derivation observed anywhere.

---

## Conclusion

All three triage points confirmed. CPU uses a single global scalar (`CalcScoreStDev` →
`scoreStDev`) computed once per tree, passed unchanged into every per-partition and
per-leaf candidate evaluation for all three grow policies (Oblivious, Depthwise,
Lossguide).

**Mechanism confirmed: global scalar; MLX implementation will mirror DEC-028's `gradRms`
threading for the non-oblivious call sites.**
