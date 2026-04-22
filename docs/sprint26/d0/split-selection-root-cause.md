# Sprint 26 D0-5: Split Selection Root Cause

**Branch:** `mlx/sprint-26-python-parity`  
**Date:** 2026-04-22  
**Probes:** P9 (config), P10 (borders), P11 (top-10 candidates), P12 (gain formula), P13 (cross-check)  
**Raw output:** `benchmarks/sprint26/d0/one-tree-instrumentation.txt`

---

## 1. Effective Config Diff (P9)

| Parameter | MLX value | CPU (equivalent) | Differs |
|---|---|---|---|
| ColsampleByTree (rsm) | 1.0 | 1.0 | no |
| SubsampleRatio | 1.0 | N/A (bootstrap=No) | — |
| RandomStrength | 1.0 (default) | 1.0 (explicit) | formula only |
| MinDataInLeaf | 1 | 1 | no |
| L2RegLambda | 3.0 | 3.0 | no |
| BootstrapType | "no" | "No" | no (string case only) |
| BaggingTemperature | 1.0 | N/A | — |
| MvsReg | 0.0 | N/A | — |
| FeatureBorderType | EqualFrequency (custom) | GreedyLogSum | yes |
| **noise_scale formula** | **rs × N** | **rs × sqrt(sum(g²)/N)** | **YES** |

**noise_scale ratio MLX/CPU = 16,896×** at iter 0, N=10000, std(y)=0.592.

No parameter was silently defaulted to < 1.0. H5 (ColsampleByTree < 1.0 by default) is falsified.

---

## 2. Border Diff Summary (P10)

MLX quantization (EqualFrequency, custom implementation):

| Feature | count | min | max |
|---|---|---|---|
| 0 | 127 | −2.421 | 2.390 |
| 1 | 127 | −2.455 | 2.414 |

CPU (GreedyLogSum) only stores borders that appear in the model's tree splits (3 per feature for a depth-6 tree). Direct full-grid comparison is not available from the public API. From the JSON model, the CPU root split for feature 0 used border −0.03267, which maps to approximately MLX bin 62 (border −0.03249). MLX's top-1 candidate is bin 60 (border −0.07300).

Both values are near-zero crossings of X[:,0]. The gains are 1602.11 (CPU's choice, rank 2 in MLX) and 1602.32 (MLX's rank-1), a 0.013% difference. The two methods quantize Gaussian data slightly differently (GreedyLogSum vs EqualFrequency) but choose functionally equivalent splits when noise is absent.

**H2 (quantization border divergence) falsified as root cause** — with RandomStrength=0, MLX achieves ratio 0.9188 vs CPU ratio 0.9182. The 0.06% gap is the border algorithm difference, not a material regression.

Full border arrays saved to `benchmarks/sprint26/d0/borders-feat01.json`.

---

## 3. Top-10 Split Candidates (P11)

At depth=0, iter=0, RandomStrength=0 (noise disabled), MLX produces these candidates sorted by true gain:

| Rank | Feature | BinId | Gain | sumL | wL | sumR | wR |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 60 | 1602.322 | 1999.86 | 4766 | −1999.86 | 5234 |
| 1 | 0 | 62 | 1602.108 | 2001.67 | 4922 | −2001.67 | 5078 |
| 2 | 0 | 63 | 1601.770 | 2001.71 | 5000 | −2001.71 | 5000 |
| 3 | 0 | 59 | 1601.309 | 1997.52 | 4688 | −1997.52 | 5312 |
| ... | | | | | | | |
| 9 | 0 | 68 | 1593.200 | 1990.24 | 5391 | −1990.24 | 4609 |

CPU's chosen root split was feature=0, border≈−0.03267 = MLX bin 62. **CPU's choice is at MLX rank 2.** The gain delta between rank-1 and rank-2 is 0.214 (0.013% relative).

With RandomStrength=1.0, MLX adds noise with σ=10,000 to each gain score. The maximum true gain in this list is 1602.32. The noise term has expected absolute magnitude ~10,000, completely overwhelming the 1602.32 signal. The tree selects a nearly random split, explaining the 0.72 prediction std ratio (vs 0.92 with noise=0).

---

## 4. Gain Formula (P12)

**MLX formula** (`csv_train.cpp:1165-1167`, ordinal path):

```
gain += (sumL²) / (wL + L2)
      + (sumR²) / (wR + L2)
      − (sumP²) / (wP + L2)
```

**CPU formula** (`greedy_tensor_search.cpp`, `score_calcers.cpp`, plain boosting):  
Identical expression with same L2 placement. For RMSE, `wL`/`wR` = doc counts = hessian sums (all hessians = 1). Parent term is present in both. No `0.5×` prefactor in either.

**Noise formula — divergence:**

| | Formula | Value at iter=0, N=10000 |
|---|---|---|
| MLX | `rs × totalWeight / (numPartitions × K)` | `1.0 × 10000 / 1 = 10000` |
| CPU | `rs × sqrt(sum(g_i²) / N)` | `1.0 × sqrt(mean(g²)) ≈ std(y) = 0.592` |

**H1 (missing parent term or wrong L2 placement) falsified.** The gain formula is correct.

---

## 5. Cross-Check (P13)

For rank-0 candidate (feature=0, bin=60):

| Quantity | Value |
|---|---|
| sumL | 1999.860 |
| wL | 4766 |
| sumR | −1999.860 |
| wR | 5234 |
| sumP | ~0.0 (near-zero; mean(g)≈0) |
| wP | 10000 |
| gain by MLX formula | 1602.322 |
| gain by CPU formula | 1602.322 (identical) |
| gain reported by FindBestSplit | 1602.322 |
| L2 | 3.0 |

All three agree. **H3 (histogram corruption) falsified.** The histogram inputs are correct.

---

## 6. Root Cause

**H5 (RandomStrength noise formula misscaled) is the root cause.**

MLX's `FindBestSplit` computes noise scale as:

```cpp
// csv_train.cpp:990
noiseScale = randomStrength * static_cast<float>(totalWeight / (numPartitions * K + 1e-10));
```

For RMSE with N=10000, K=1, numPartitions=1: `noiseScale = 1.0 × 10000`.

CPU CatBoost uses `CalcDerivativesStDevFromZeroPlainBoosting`, which computes:

```cpp
// greedy_tensor_search.cpp:92-107
return sqrt(sum2 / weightedDerivatives.front().size());
// = sqrt(sum(g_i²) / N) = gradient RMS
```

For RMSE at iter 0: `noiseScale = 1.0 × sqrt(mean(g²)) ≈ std(y) ≈ 0.592`.

**Ratio: 10000 / 0.592 = 16,896×.** The noise completely drowns the true gain signal (best gain 1602 vs noise σ=10,000). The tree selects a near-random split at every depth level.

All other hypotheses are falsified:

| Hypothesis | Status | Evidence |
|---|---|---|
| H5: noise formula misscaled | **ROOT CAUSE** | P9: ratio 16,896×; MLX rs=0 gives 0.9188 == CPU rs=0 |
| H1: gain formula wrong | Falsified | P12/P13: exact match, parent term present |
| H2: border divergence | Falsified | P10+P11: 0.06% gap, rank-2 choice with rs=0 |
| H3: histogram corruption | Falsified | P13: all three gain values agree |
| H4: RandomStrength/MinDataInLeaf default | Falsified | P9: all defaults match |
| H6: goLeft/goRight flipped | Falsified | P4: partitions.max()=63 exactly |

---

## 7. Proposed Fix Direction

**File:** `catboost/mlx/tests/csv_train.cpp:980-991`

Replace the current noise-scale computation:

```cpp
// CURRENT (wrong — uses hessian sum, not gradient RMS)
float noiseScale = 0.0f;
if (randomStrength > 0.0f && rng) {
    double totalWeight = 0.0;
    for (ui32 p = 0; p < numPartitions; ++p)
        for (ui32 k = 0; k < K; ++k)
            totalWeight += std::abs(perDimPartStats[k][p].Weight);
    noiseScale = randomStrength * static_cast<float>(totalWeight / (numPartitions * K + 1e-10));
}
```

With the CPU-matching formula:

```cpp
// PROPOSED (matches CPU: rs × sqrt(sum(g²)/N) = rs × gradient_RMS)
float noiseScale = 0.0f;
if (randomStrength > 0.0f && rng) {
    // CPU formula: CalcDerivativesStDevFromZeroPlainBoosting
    //   = sqrt(sum(g_i^2) / N)  [gradient RMS, not hessian mean]
    // perDimHist[k] stores per-bin gradient/hessian sums; recover sum(g^2)
    // from the partition-level stats: perDimPartStats[k][p] has Sum=grad_sum,
    // Weight=hess_sum. For sum(g^2) we need the full per-doc gradient vector,
    // which is NOT available in FindBestSplit. Pass gradRms as a new parameter,
    // or compute it at the call site in RunTraining before each depth level.
    // Simplest: add float gradRms parameter with default -1 (auto-compute below).
}
```

The cleanest implementation: compute `gradRms` once per iteration in `RunTraining` right after gradient computation (it's just `sqrt(mean(g^2))` over `dimGrads[0]`), then pass it as a new parameter to `FindBestSplit`. This requires a one-line addition to `RunTraining` and a parameter change to `FindBestSplit`/`FindBestSplitPerPartition`.

**Do not apply the fix in this commit — document only per D0-6 protocol.**
