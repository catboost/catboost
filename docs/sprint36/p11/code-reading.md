# S36-P11 T0b — Code Reading: Hessian vs Sample-Weight in Histogram Stat Slot

Branch: `mlx/sprint-36-hess-vs-weight` | Commit: `3bc02bc3cb`
Task: Read-only investigation. No source files modified.

---

## Question 1 — MLX histogram weight slot population

**Short answer:** MLX accumulates `Σ dimHess[k][doc]` into stat-slot-1 — the per-document second-order gradient (hessian). It does not accumulate sample weights.

### Metal kernel (stat-slot dispatch)

`catboost/mlx/kernels/hist.metal:113-126`

The Metal kernel is fully generic. It receives one flat `stats` buffer and reads it as:

```
const float stat = stats[statIdx * totalNumDocs + docIdx];
```

`statIdx` iterates over the third dimension of the GPU grid (`threadgroup_position_in_grid.z`).
When the host dispatches with `numStats = 2`, `statIdx=0` produces the gradient histogram and `statIdx=1` produces the "weight" histogram. The kernel has no knowledge of what the content of `stats` is.

### Host-side dispatch: what gets concatenated into stat-slot-1

`catboost/mlx/methods/histogram.cpp:269-279` (overload used by the training loop):

```cpp
auto statsArr = mx::concatenate({
    mx::reshape(dataset.GetGradients(), {1, static_cast<int>(numDocs)}),
    mx::reshape(dataset.GetHessians(), {1, static_cast<int>(numDocs)})
}, 0);
```

`GetHessians()` returns the array stored in the MLX dataset object, which is `dimHess[k]` after the
derivative computation step.

### Training loop: how statsK is built before the histogram call

Lossguide path: `catboost/mlx/tests/csv_train.cpp:4374-4383`

```cpp
auto statsK = mx::concatenate({
    mx::reshape(dimGrads[k], {1, static_cast<int>(trainDocs)}),
    mx::reshape(dimHess[k],  {1, static_cast<int>(trainDocs)})
}, 0);
statsK = mx::reshape(statsK, {static_cast<int>(2 * trainDocs)});
// ...
histArrays2.push_back(DispatchHistogram(compressedData, statsK, ...));
```

Depthwise/SymmetricTree path: `catboost/mlx/tests/csv_train.cpp:4558-4568`

```cpp
auto statsK = mx::concatenate({
    mx::reshape(dimGrads[k], {1, static_cast<int>(trainDocs)}),
    mx::reshape(dimHess[k], {1, static_cast<int>(trainDocs)})
}, 0);
statsK = mx::reshape(statsK, {static_cast<int>(2 * trainDocs)});
histArrays.push_back(DispatchHistogram(compressedData, statsK, ...));
```

In both paths the second "row" (stat-slot-1) is `dimHess[k]` — the loss hessian.

### What `dimHess[k]` actually is, per loss function

`catboost/mlx/tests/csv_train.cpp:4067-4160`:

| Loss | `dimHess[0]` |
|------|-------------|
| rmse | `ones(trainDocs)` — constant 1 |
| mae, quantile | `ones(trainDocs)` — constant 1 |
| huber | `1.0` (abs diff ≤ δ) or `1e-6` (otherwise) |
| logloss | `max(σ(f)·(1−σ(f)), 1e-16)` — sigmoid variance |
| poisson | `max(exp(f), 1e-6)` — predicted rate |
| tweedie | compound of exp terms |
| mape | `1 / max(|target|, 1e-6)` |
| multiclass | `max(p_k·(1−p_k), 1e-16)` per class |

For losses other than RMSE/MAE/quantile, `dimHess[k]` is not 1 and is not the sample weight.

### Effect of sample weights

`catboost/mlx/tests/csv_train.cpp:4164-4172`:

```cpp
if (!sampleWeights.empty()) {
    auto sampleWeightsArr = mx::array(sampleWeights.data(), ...);
    for (ui32 k = 0; k < approxDim; ++k) {
        dimGrads[k] = mx::multiply(dimGrads[k], sampleWeightsArr);
        dimHess[k]  = mx::multiply(dimHess[k], sampleWeightsArr);
    }
}
```

When sample weights are provided, `dimHess[k]` becomes `hessian_k[doc] * sampleWeight[doc]`.
The stat-slot-1 histogram then accumulates `Σ (hess[doc] * sampleWeight[doc])` — a product of both, not a pure sample weight sum.

**Conclusion for Q1:** MLX's stat-slot-1 accumulates `Σ dimHess[k][doc]` (possibly scaled by sample weights when provided, but still driven by the loss hessian, not a raw document count or pure sample weight).

---

## Question 2 — CPU equivalent (`SumWeight` in scoring.cpp)

**Short answer:** CPU's `SumWeight` is `Σ sampleWeights[doc]` — the per-document sample weight, not the hessian.

### Accumulation site

`catboost/private/libs/algo/scoring.cpp:203-224` (`UpdateWeighted`):

```cpp
for (int doc : docIndexRange.Iter()) {
    auto& leafStats0 = stats[indexer.GetIndex<isOneNode>(doc, quantizedValues)];
    leafStats0.SumWeightedDelta += weightedDer[doc];
    leafStats0.SumWeight += sampleWeights[doc];    // line 217
}
```

`sampleWeights` here is `sampleWeightsData`, assigned at line 278-279:

```cpp
const float* sampleWeightsData = hasPairwiseWeights ?
    GetDataPtr(bt.SamplePairwiseWeights) : GetDataPtr(fold.SampleWeights);
```

### Where `fold.SampleWeights` comes from

`catboost/private/libs/algo/fold.cpp:119`:

```cpp
ff.SampleWeights.resize(learnSampleCount, 1);
```

It is initialised to 1.0 for all documents, then `SetWeights` is called:

`catboost/private/libs/algo/fold.cpp:213-219`:

```cpp
void TFold::SetWeights(TConstArrayRef<float> weights, ui32 learnSampleCount) {
    if (!weights.empty()) {
        AssignPermuted(weights, &LearnWeights);
        SumWeight = Accumulate(weights.begin(), weights.end(), (double)0.0);
    } else {
        SumWeight = learnSampleCount;
    }
}
```

`SampleWeights` (the per-document bootstrap weighting vector) stays at 1.0 unless bootstrap modifies it. The user-supplied `weights=` parameter goes into `LearnWeights` (used by the `UpdateDeltaCount` path), not `SampleWeights`. For plain boosting (non-ordered), `isPlainMode=true`, so the `UpdateWeighted` path runs and uses `SampleWeights`.

### For Logloss specifically

CPU computes `SumWeightedDelta += weightedDer[doc]` where `weightedDer = CalcDer(Logloss) * sampleWeight`. `CalcDer(Logloss)` is the gradient `σ(f) − y`, pre-multiplied by `sampleWeight`.
CPU's score denominator via `AddLeafPlain` (`score_calcers.cpp:20-33`) is `CalcAverage(SumWeightedDelta, SumWeight, L2)`. The `SumWeight` in the denominator is the sum of sample weights, not the sum of `σ(1−σ)` hessians.

In contrast, MLX uses `dimHess[0] = max(σ(f)·(1−σ(f)), 1e-16)` as the per-doc weight in its score denominator (`csv_train.cpp:4094-4097`, then at `csv_train.cpp:1895-1897` via `suffHess`).

**The F11 claim is correct:** CPU uses `Σ sampleWeights[doc]` and MLX uses `Σ dimHess[k][doc]`. For Logloss, these are `Σ w[doc]` vs `Σ σ(1−σ)[doc]` — categorically different quantities.

---

## Question 3 — RMSE-only invariance check

### RMSE hessian is constant 1

`catboost/mlx/tests/csv_train.cpp:4070`:

```cpp
if (lossType == "rmse") {
    dimHess[0] = mx::ones({static_cast<int>(trainDocs)}, mx::float32);
```

When no sample weights are provided (`sampleWeights.empty()` is true), the multiplication block at
`csv_train.cpp:4165-4172` does not execute. `dimHess[0]` stays as `ones(trainDocs)`.

The histogram stat-slot-1 therefore accumulates `Σ 1 = docCount` per bin, which equals `Σ sampleWeights[doc]` when all sample weights are 1.0 — as CPU initialises them at `fold.cpp:119`.

Both sides produce `docCount` in the weight/hessian slot for unweighted RMSE. The invariance holds.

### Bootstrap weights

`catboost/mlx/tests/csv_train.cpp:4234-4239`: when `useBootstrap=true`, bootstrap weights are also
multiplied into `dimHess[k]`, making `dimHess[k] = hess * bootstrapWeight`. For RMSE, `hess=1`, so
`dimHess[k] = bootstrapWeight`. CPU similarly applies bootstrap weights into `SampleWeights`. The
invariance extends to bootstrapped RMSE.

---

## Question 4 — Newton update at leaf time

**Short answer:** MLX uses the same `dimHess[k]` array for both the score-function denominator (via histogram stat-slot-1) and the Newton leaf update denominator. The two are not independent.

### Leaf value computation

`catboost/mlx/tests/csv_train.cpp:5036-5049` (approxDim == 1):

```cpp
auto gSumsArr = mx::scatter_add_axis(leafTarget, partitions, dimGrads[0], 0);
auto hSumsArr = mx::scatter_add_axis(leafTarget, partitions, dimHess[0], 0);
leafValues = mx::negative(mx::multiply(lrArr,
    mx::divide(gSumsArr, mx::add(hSumsArr, l2Arr))));
```

The Newton step is: `leafValue = −lr · ΣG / (ΣH + λ)`.

`gSumsArr` = per-leaf sum of `dimGrads[0]` (gradient sum).
`hSumsArr` = per-leaf sum of `dimHess[0]` (hessian sum — the same array used in histogram stat-slot-1).

`catboost/mlx/tests/csv_train.cpp:5195-5200` (multiclass / approxDim > 1):

```cpp
auto gSumsArr = mx::scatter_add_axis(leafTarget, partitions, dimGrads[k], 0);
auto hSumsArr = mx::scatter_add_axis(leafTarget, partitions, dimHess[k], 0);
dimLeafVals.push_back(mx::negative(mx::multiply(lrArr,
    mx::divide(gSumsArr, mx::add(hSumsArr, l2Arr)))));
```

Same pattern for all K dimensions.

### Are the same arrays used for both purposes?

Yes. `dimHess[k]` is the single array that appears in:
1. `statsK` concatenation for histogram stat-slot-1 (score function denominator, `csv_train.cpp:4376, 4560`)
2. `hSumsArr` scatter_add for Newton denominator (`csv_train.cpp:5037, 5196`)

There is no separate "weight" array distinct from the hessian. Any fix to the score-function denominator
that substitutes a different array (e.g., sample weights) would need to either:
(a) also change the Newton denominator (potentially breaking Newton step for RMSE/loss types where hessian
is the correct denominator for Newton), or
(b) maintain two separate arrays: one for the score function and one for the Newton step.

This is the entanglement the task description anticipated: the bug is structurally unified — the same
`dimHess[k]` drives both the split-scoring denominator and the leaf-value denominator.

### CPU leaf value (for comparison)

CPU uses `SumWeights` from the fold's `TSum` structure for Newton leaf values
(`online_predictor.h:145`: `CalcAverage(ss.SumDer, ss.SumWeights, ...)`). For plain boosting without
sample weights, `SumWeights = Σ sampleWeights = Σ 1 = docCount`. For Logloss with sample weights,
`SumWeights = Σ sampleWeight[doc]` — not the hessian. CPU uses the same `SumWeights` for both score
function denominator and leaf value denominator. So the entanglement is symmetric: CPU also uses one
quantity for both purposes, it just uses sample weights whereas MLX uses hessians.

---

## Question 5 — Score function consumer: tracing `weightLeft`

### One-hot path (L2)

`catboost/mlx/tests/csv_train.cpp:1690-1712`:

```cpp
float totalSum    = perDimPartStats[k][p].Sum;
float totalWeight = perDimPartStats[k][p].Weight;    // = hSPtr[p]

float sumRight    = histData[feat.FirstFoldIndex + bin];
float weightRight = histData[totalBinFeatures + feat.FirstFoldIndex + bin];   // line 1694
float sumLeft     = totalSum - sumRight;
float weightLeft  = totalWeight - weightRight;
```

`histData[totalBinFeatures + feat.FirstFoldIndex + bin]` reads from the second stat-plane of the
histogram — exactly what stat-slot-1 contains (`dimHess[k]` per-bin sums).

`perDimPartStats[k][p].Weight` is set at `csv_train.cpp:4638`:

```cpp
perDimPartStats[k][p].Weight = hsPtr[p];
```

where `hsPtr` is the data pointer from `hessSumArrays[k]` — the scatter_add of `dimHess[k]` over
partitions.

### Ordinal path (L2 + Cosine)

`catboost/mlx/tests/csv_train.cpp:1791-1801`:

```cpp
std::vector<float> suffHess(K * numPartitions * stride, 0.0f);
// ...
for (int b = ...; b >= 0; --b) {
    suffHess[base + b] = suffHess[base + b + 1]
                       + hd[totalBinFeatures + feat.FirstFoldIndex + b];   // line 1801
}
```

`hd` is `perDimHist[k].data() + p * 2 * totalBinFeatures`. The offset `totalBinFeatures + ...` again
selects the second stat-plane. The suffix sum of the second stat-plane produces `suffHess`.

`csv_train.cpp:1895-1897`:

```cpp
float weightRight = suffHess[base + bin];
float sumLeft     = totalSum - sumRight;
float weightLeft  = totalWeight - weightRight;
```

`suffHess[base + bin]` = suffix sum of `dimHess[k]` bins from `bin` to `folds-1`.

### Semantic summary for `weightLeft`

`weightLeft` is `totalHessianSum_in_partition − suffix_hessian_sum_from_bin`, which equals the prefix
hessian sum in the left child (bins 0..bin-1). It is not labelled as such anywhere in the code — the
variable name "weightLeft" is historically borrowed from the CatBoost CPU naming convention where
`SumWeight` played the same structural role (denominator in Newton / CalcAverage), even though the
actual quantity stored differs between CPU and MLX.

---

## Question 6 — Naming check: what each variable actually is

| Variable / slot | File:line | What it IS | What it is NOT |
|-----------------|-----------|-----------|----------------|
| `dimHess[k]` (RMSE) | `csv_train.cpp:4070` | Per-doc constant 1.0f | Hessian (trivially 1) |
| `dimHess[k]` (Logloss) | `csv_train.cpp:4094-4097` | `max(σ·(1−σ), 1e-16)` — 2nd derivative of log-loss | Sample weight |
| `dimHess[k]` (Poisson) | `csv_train.cpp:4102` | `max(exp(f), 1e-6)` — 2nd derivative of Poisson NLL | Sample weight |
| `dimHess[k]` (with sample weights) | `csv_train.cpp:4169` | `hess[doc] * sampleWeight[doc]` — product | Pure sample weight |
| `dimHess[k]` (with bootstrap) | `csv_train.cpp:4237` | `hess[doc] * bootstrapWeight[doc]` — product | Pure sample weight |
| `statsK` stat-slot-1 | `csv_train.cpp:4376, 4560` | `dimHess[k]` flattened — see above | Sample weight sum |
| `suffHess[base+bin]` | `csv_train.cpp:1801` | Suffix sum of `dimHess[k]` bins | Suffix sum of sample weights |
| `weightRight` / `weightLeft` | `csv_train.cpp:1694-1696, 1895-1897` | Hessian sum in right/left child | Sample weight sum |
| `perDimPartStats[k][p].Weight` | `csv_train.cpp:4638` | Total hessian sum in partition p, dim k | Sample weight count |
| `hSumsArr` (Newton) | `csv_train.cpp:5037, 5196` | Per-leaf hessian sum (scatter_add of dimHess) | Per-leaf sample weight sum |
| CPU `SumWeight` | `scoring.cpp:217` | `Σ sampleWeights[doc]` in bin-stat bucket | Hessian |
| CPU `fold.SampleWeights` | `fold.cpp:119` | 1.0 per doc (default), bootstrap weight otherwise | Hessian |
| CPU `leafStats.SumWeightedDelta` | `score_calcers.h:64,88` | `Σ (gradient * sampleWeight)` | Pure gradient |

---

## Surprises

### Surprise 1: The F11 line numbers have drifted

F11 cites `csv_train.cpp:3780, 3967` for the MLX histogram construction. Current master tip has these
calls at lines 4374-4383 and 4558-4568. The line numbers shifted by roughly +600 lines due to code added
in Sprints 29-35 (snapshot/resume, Lossguide, bootstrap, debug instrumentation). The _logic_ is unchanged.

### Surprise 2: `perDimPartStats[k][p].Weight` is NOT read from the histogram

The partition-level totals `perDimPartStats[k][p].Sum` and `.Weight` are NOT extracted from the histogram.
They come from separate `scatter_add_axis` calls:

`csv_train.cpp:4570-4575`:

```cpp
gradSumArrays.push_back(mx::scatter_add_axis(
    mx::zeros(...), partitions, dimGrads[k], 0));
hessSumArrays.push_back(mx::scatter_add_axis(
    mx::zeros(...), partitions, dimHess[k], 0));
```

and read back at `csv_train.cpp:4634-4638`:

```cpp
perDimPartStats[k][p].Sum    = gsPtr[p];
perDimPartStats[k][p].Weight = hsPtr[p];
```

This means the "total weight" used for `totalWeight - weightRight = weightLeft` is computed
independently from the histogram — but is still `Σ dimHess[k]` over partition p, not a sample weight
count. The two paths (histogram second plane, scatter_add) are consistent: both accumulate `dimHess[k]`.

### Surprise 3: No code path puts raw sample weights into the score denominator

At no point does MLX pass the raw `sampleWeights` vector (from the CSV `--sample-weights` input) as a
standalone stat to the histogram or the score function. Sample weights only enter via multiplication
into `dimHess[k]` and `dimGrads[k]`. There is no branch that substitutes the sample weight for the
hessian in the denominator. The F11 claim correctly identifies this as a structural missing path.

### Surprise 4: CPU also uses the SAME quantity for score denominator and Newton denominator

F11's framing ("MLX plugs hessian where CPU plugs weight") might imply the fix is simply to swap the
MLX denominator to sample weights. But CPU also uses `SumWeight` (its sample-weight-based quantity) for
BOTH the score function (`scoring.cpp:217` → `AddLeafPlain`) AND the Newton leaf update
(`online_predictor.h:117`: `CalcAverage(SumDer, SumWeight, ...)`). CPU does not use hessians anywhere
in its score/Newton pipeline. MLX uses hessians in both places. The symmetry is exact. A fix that wants
to match CPU must replace `dimHess[k]` with a sample-weight array in both the histogram stat-slot and
the Newton scatter_add.

### Surprise 5: For RMSE with no sample weights, the invariance is fully symmetric

CPU: `SumWeight = Σ 1 = docCount` (from `fold.cpp:119` initialising to 1).
MLX: `Σ dimHess[k] = Σ 1 = docCount` (from `csv_train.cpp:4070` setting `dimHess[0] = ones`).
These are numerically identical for all bins and partitions. F11 is correct that the bug is latent and
does not fire at the RMSE anchor.

---

## Citation index (file:line)

| Claim | File:line |
|-------|-----------|
| Metal kernel stat-slot read | `catboost/mlx/kernels/hist.metal:115` |
| Host: grad+hess concatenation into statsArr | `catboost/mlx/methods/histogram.cpp:270-276` |
| Training loop: statsK lossguide | `catboost/mlx/tests/csv_train.cpp:4374-4383` |
| Training loop: statsK depthwise | `catboost/mlx/tests/csv_train.cpp:4558-4568` |
| dimHess RMSE = ones | `catboost/mlx/tests/csv_train.cpp:4070` |
| dimHess Logloss = sigmoid variance | `catboost/mlx/tests/csv_train.cpp:4094-4097` |
| dimHess Poisson = exp(f) | `catboost/mlx/tests/csv_train.cpp:4102` |
| Sample weight multiplication into dimHess | `catboost/mlx/tests/csv_train.cpp:4169` |
| Bootstrap weight multiplication into dimHess | `catboost/mlx/tests/csv_train.cpp:4237` |
| partStats Weight from hessSumArrays | `catboost/mlx/tests/csv_train.cpp:4638` |
| suffHess from histogram second plane | `catboost/mlx/tests/csv_train.cpp:1801` |
| weightRight from suffHess | `catboost/mlx/tests/csv_train.cpp:1895` |
| Newton: gSumsArr / hSumsArr from dimGrads / dimHess | `catboost/mlx/tests/csv_train.cpp:5036-5037` |
| Newton: multiclass path | `catboost/mlx/tests/csv_train.cpp:5195-5196` |
| CPU UpdateWeighted: SumWeight += sampleWeights[doc] | `catboost/private/libs/algo/scoring.cpp:217` |
| CPU sampleWeightsData source | `catboost/private/libs/algo/scoring.cpp:278-279` |
| CPU fold SampleWeights init to 1 | `catboost/private/libs/algo/fold.cpp:119` |
| CPU CalcAverage denominator = count | `catboost/private/libs/algo_helpers/online_predictor.h:117` |
| CPU L2 AddLeafPlain uses SumWeight | `catboost/private/libs/algo/score_calcers.cpp:21-30` |
| CPU Cosine AddLeafPlain via UpdateScoreBinKernelPlain | `catboost/private/libs/algo/score_calcers.cpp:11` |
| CPU Cosine denominator = sumWeight (generic path) | `catboost/libs/helpers/short_vector_ops.h:77-80` |
