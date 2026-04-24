# S30-D1-CPU-COSINE-AUDIT — Verdict

**VERDICT: CPU CatBoost Cosine is fp64 (double) at every layer L1–L5. There is no fp32 anywhere in the gain/score/argmax/leaf-value path. The gradients themselves enter the path as `double`.**

---

## 1. Scope

Audit target: CPU CatBoost Cosine score path used by the `RMSE + SymmetricTree + Cosine` configuration (the same config MLX is being measured against in S30). All file:line citations are against the working tree at branch `mlx/sprint-30-cosine-kahan` (HEAD `b0c853a6f6`).

The dispatcher confirms that for `EScoreFunction::Cosine` the CPU path lands in `TCosineScoreCalcer` (not in any templated/SIMD-only class):

```cpp
// catboost/private/libs/algo/score_calcers.h:106-115
inline THolder<IPointwiseScoreCalcer> MakePointwiseScoreCalcer(EScoreFunction scoreFunction) {
    switch(scoreFunction) {
        case EScoreFunction::Cosine:
            return MakeHolder<TCosineScoreCalcer>();
        ...
    }
}
```

For `SymmetricTree`, scoring goes through `scoring.cpp::CalcStatsKernel` → `UpdateWeighted` → `TCosineScoreCalcer::AddLeafPlain` → `NSimdOps::UpdateScoreBinKernelPlain` → `GetScores()`. For the lossguide/leafwise variant the same `TCosineScoreCalcer` is used via `leafwise_scoring.cpp`. Both surfaces are inspected below; both are fp64 end-to-end.

---

## 2. Per-Layer Evidence Table

| Layer | Type used | File:Line | Excerpt |
|-------|-----------|-----------|---------|
| **L1 — partition sums** (gradient & weight per-doc accumulation into the bucket histogram) | **fp64** | `catboost/private/libs/algo/calc_score_cache.h:72-77` | `struct TBucketStats { double SumWeightedDelta; double SumWeight; double SumDelta; double Count; };` |
| L1 — accumulation kernel (oblivious / SymmetricTree path) | **fp64** | `catboost/private/libs/algo/scoring.cpp:203-218` | `inline static void UpdateWeighted(...const double* weightedDer, ...) { ... leafStats0.SumWeightedDelta += weightedDer[doc]; leafStats0.SumWeight += sampleWeights[doc]; }` |
| L1 — accumulation kernel (leafwise path, identical pattern) | **fp64** | `catboost/private/libs/algo/leafwise_scoring.cpp:226-242` | `inline void UpdateWeighted(...const double* weightedDer, ... TBucketStats* stats) { ... leafStats.SumWeightedDelta += weightedDer[doc]; leafStats.SumWeight += sampleWeights[doc]; }` |
| L1 — gradient source (so the `double*` above is genuine fp64, not a widened fp32) | **fp64** | `catboost/private/libs/algo/tensor_search_helpers.cpp:467-472` and `:580` | `TVector<TVector<double>>* weightedDerivatives = &bt.WeightedDerivatives;` and `sampleWeightedDerivativesData[z] = weightedDerivativesData[z] * sampleWeightsData[z];` |
| **L2 — per-term arithmetic** (the `sL²·invL + sR²·invR` style scalar formula, generic path) | **fp64** | `catboost/libs/helpers/short_vector_ops.h:17` + `:61-81` | `using TValueType = double;` ... `void UpdateScoreBinKernelPlain(double scaledL2Regularizer, const NSimdOps::TValueType* trueStatsPtr, ... NSimdOps::TValueType* scoreBinPtr) { const double trueAvrg = CalcAverage(trueStatsPtr[0], trueStatsPtr[1], scaledL2Regularizer); ... scoreBinPtr[0] += trueAvrg * trueStatsPtr[0]; scoreBinPtr[1] += trueAvrg * trueAvrg * trueStatsPtr[1]; ...}` |
| L2 — same op, SSE2 specialization actually used on x86 builds | **fp64** | `catboost/libs/helpers/short_vector_ops.h:108` + `:155-175` | `static_assert(std::is_same<NSimdOps::TValueType, double>::value, "NSimdOps::TValueType must be double");` ... `const __m128d trueStats = _mm_loadu_pd(trueStatsPtr); ... const __m128d average = _mm_mul_pd(sumWeightedDelta, _mm_and_pd(isSumWeightPositive, _mm_div_pd(_mm_set1_pd(1.0), _mm_add_pd(sumWeight, regularizer))));` (every intrinsic is `_pd` = packed double) |
| L2 — TCosineScoreCalcer wrapper that calls the SIMD op | **fp64** | `catboost/private/libs/algo/score_calcers.cpp:10-12` | `void TCosineScoreCalcer::AddLeafPlain(int splitIdx, const TBucketStats& leftStats, const TBucketStats& rightStats) { NSimdOps::UpdateScoreBinKernelPlain(L2Regularizer, &rightStats.SumWeightedDelta, &leftStats.SumWeightedDelta, &Scores[splitIdx][0]); }` |
| L2 — Cosine accumulator container | **fp64** | `catboost/private/libs/algo/score_calcers.h:47-53` | `class TCosineScoreCalcer final : public IPointwiseScoreCalcer { using TFraction = std::array<double, 2>; ... Scores.resize(splitsCount, {0, 1e-100}); }` |
| **L3 — gain cast** (the value handed back to the search) | **fp64** | `catboost/private/libs/algo/score_calcers.h:55-61` | `TVector<double> GetScores() const override { TVector<double> scores(SplitsCount); for (int i : xrange(SplitsCount)) { scores[i] = Scores[i][0] / sqrt(Scores[i][1]); } return scores; }` |
| L3 — virtual contract that pins this to `double` for every score function | **fp64** | `catboost/private/libs/algo/score_calcers.h:25` | `virtual TVector<double> GetScores() const = 0;` |
| L3 — score then carried in `TRandomScore::Val` (the per-candidate "best score" stored on each candidate) | **fp64** | `catboost/private/libs/algo/rand_score.h:19-22` and `:42-44` | `struct TRandomScore { ERandomScoreDistribution Distribution; double Val; double StDev; ... template <typename TRng> double GetInstance(TRng& rand) const { if (...Normal) return Val + NormalDistribution<double>(rand, 0, StDev); ... }` |
| L3 — score-storage handoff from `GetScores()` into the candidate (no narrowing) | **fp64** | `catboost/private/libs/algo/tensor_search_helpers.cpp` `SetBestScore`, `:716-739` | `void SetBestScore(..., const TVector<TVector<double>>& allScores, ..., TVector<TCandidateInfo>* subcandidates) { ... double bestScoreInstance = MINIMAL_SCORE; ... const double scoreWoNoise = scores[binFeatureIdx]; TRandomScore randomScore(scoreDistribution, scoreWoNoise, scoreStDev); const double scoreInstance = randomScore.GetInstance(rand); if (scoreInstance > bestScoreInstance) { ...; subcandidateInfo.BestScore = std::move(randomScore); ...}}` |
| **L4 — argmax across candidates** | **fp64** | `catboost/private/libs/algo/greedy_tensor_search.cpp:941-966` (function `SelectBestCandidate`) | `double bestGain = -std::numeric_limits<double>::infinity(); ... double score = candidate.BestScore.GetInstance(ctx.LearnProgress->Rand); ... double gain = score - scoreBeforeSplit; ... if (gain > bestGain) { *bestScore = score; *bestSplitCandidate = &candidate; bestGain = gain; }` (note: `MINIMAL_SCORE = std::numeric_limits<double>::lowest()` per `rand_score.h:12`) |
| **L5 — leaf-value sum** (per-leaf gradient/hessian accumulator) | **fp64** | `catboost/private/libs/algo_helpers/online_predictor.h:13-16` | `struct TSum { double SumDer = 0.0; double SumDer2 = 0.0; double SumWeights = 0.0; ... };` |
| L5 — per-doc derivative source | **fp64** | `catboost/private/libs/algo_helpers/ders_holder.h:3-7` | `struct TDers { double Der1; double Der2; double Der3; };` |
| L5 — Newton step divisor itself | **fp64** | `catboost/private/libs/algo_helpers/online_predictor.h:162-179` | `inline double CalcDeltaNewtonBody(double sumDer, double sumDer2, float l2Regularizer, double sumAllWeights, int allDocCount) { return sumDer / (-sumDer2 + l2Regularizer * (sumAllWeights / allDocCount)); }` |
| L5 — Newton leaf-delta application (callers) | **fp64** | `catboost/private/libs/algo/approx_calcer.cpp:541-548` | `(*leafDeltas)[leafIndex] = leafDers[leafIndex].SumDer / leafWeight;` (with `leafWeight = -leafDers[leafIndex].SumDer2 + scaledL2Regularizer;` — both `double`) |

---

## 3. Divergence Summary — CPU vs MLX (post-T2/K4)

T2 verdict (`docs/sprint30/t2-kahan/verdict.md`) reports MLX's K4 patch raised the four `cosNum`/`cosDen` accumulators in `csv_train.cpp` from float32 to double, with the final gain still cast to `float` for split comparison:

```cpp
// catboost/mlx/tests/csv_train.cpp:1239-1300, :1381-1482, :1652-1692, :1735…
double cosNum_d = 0.0;
double cosDen_d = 1e-20;
...
cosNum_d += dSL * dSL * dInvL + dSR * dSR * dInvR;
cosDen_d += dSL * dSL * dWL * dInvL * dInvL + ...;
...
totalGain = static_cast<float>(cosNum_d / std::sqrt(cosDen_d));
```

Layer-by-layer divergence table:

| Layer | CPU | MLX (post-T2/K4) | Delta |
|-------|-----|-------------------|-------|
| L1 — gradient source | fp64 (`bt.WeightedDerivatives`) | **fp32** (Metal histogram kernel produces `float` per-bin sums; `perDimHist` is `std::vector<std::vector<float>>` per `csv_train.cpp:1167`) | **DIVERGE — MLX upstream is fp32** |
| L1 — partition sums going into the gain formula | fp64 | fp32 (sums stored in float histogram, then cast to double via `dSL = static_cast<double>(sumLeft)` per K4) | **DIVERGE — values were already rounded to fp32 before K4 gets to widen them** |
| L2 — per-term arithmetic | fp64 (`UpdateScoreBinKernelPlain` is all `__m128d` / `double`) | fp64 since K4 (each `sL*sL*invL + sR*sR*invR` widens to double before multiplying) | MATCH (post-K4) |
| L3 — gain scalar stored / compared | fp64 (`TVector<double> GetScores()`, `TRandomScore::Val` is `double`) | **fp32** (`totalGain = static_cast<float>(cosNum_d / std::sqrt(cosDen_d))`, `bestGain` is `float` per `csv_train.cpp:1187`) | **DIVERGE — MLX narrows to float for split comparison** |
| L4 — argmax | fp64 (`double bestGain`, `double score`, `double gain`) | fp32 (`bestGain` and per-candidate `gain` are `float`) | **DIVERGE — MLX is fp32** |
| L5 — leaf-value sum / Newton step | fp64 (`TSum` all `double`, `CalcDeltaNewtonBody` is `double`) | needs separate audit; not patched by K4 (T1 deemed Newton-step attenuation suppresses the gSum error to <4e-8 per T2 verdict §6) | UNCLEAR — out of T2 scope; likely fp32 inputs but small effective error |

**Three places MLX is still narrower than CPU:**

1. **L1 source.** The histogram itself — produced by the Metal kernel — is fp32. K4 widens *after* the bin sum has already been rounded into a 32-bit number. CPU never has this rounding step; gradients live as `double` from `CalcWeightedDerivatives` straight into `TBucketStats::SumWeightedDelta`.
2. **L3 cast.** MLX casts the gain back to `float` immediately after computing it in double (`static_cast<float>(cosNum_d / std::sqrt(cosDen_d))`). CPU never narrows: `GetScores()` returns `TVector<double>` and the value rides through `TRandomScore::Val` and `SelectBestCandidate` as `double`.
3. **L4 argmax.** Because L3 narrowed, `bestGain` in the MLX `FindBestSplit` loop is `float`. CPU's argmax is `double bestGain`.

---

## 4. Implication for S30 Scope

Q1 is now answered with high confidence: **CPU Cosine is fp64 throughout L1–L5; MLX is fp64 only at L2 (since K4) and remains fp32 at L1 (histogram source), L3 (gain cast), and L4 (argmax).** The T3 failure is therefore consistent with — and probably caused by — these three remaining narrow points, not with K4 being insufficient at L2.

This makes a **trivial-ish fix available at L3/L4**: drop the `static_cast<float>` in the four K4 sites, change `bestGain` (and the per-candidate `gain` locals) in `FindBestSplit` / `FindBestSplitPerPartition` to `double`, and propagate that through the `TBestSplitProperties` return type. That alone should close the L3/L4 gap and bit-match CPU once the histogram itself is also widened.

**L1, however, is not trivial.** The histogram is produced on the GPU in fp32. To match CPU bit-for-bit, the Metal histogram kernel would need either fp64 atomics (Metal Shading Language does not support double-precision atomics on Apple Silicon) or a redesigned reduction (e.g., split fp32 histogram into a high/low pair of accumulators à la Kahan/Klein on the host, or accumulate per-thread partial sums into double on the CPU after kernel completion). Until L1 is widened, exact CPU parity is not reachable; only the L3/L4 quantization noise can be eliminated.

**Recommendation for the parity gate**: reframe the gate as a tolerance on the *gain residual* expected from the L1 fp32 histogram quantization (Higham bound ≈ N·ε for N docs per partition), rather than bit-equality. If the iter-50 53%/27%/45% miss exceeds that bound, then either (a) the L1 fp32 histogram error compounds non-trivially across iterations through the cursor update, or (b) there is a fourth divergence not visible in this audit (most likely in L5 or in the Approx update path that consumes the leaf delta). The next step before any further patching should be a head-to-head residual decomposition at iter=1 vs iter=50 to confirm whether the drift is L1-bound or amplifies via approx-update feedback.

---

## 5. Caveats

- The audit covered the SymmetricTree (`scoring.cpp`) and Lossguide (`leafwise_scoring.cpp`) paths only. The Pairwise paths (`pairwise_scoring.cpp`) were not audited; they are not used by `RMSE + SymmetricTree + Cosine`.
- The Cython/Python entry layer was not audited but cannot affect precision since it sits above `TCosineScoreCalcer`.
- L5 on the MLX side was not measured here; the T2 verdict's claim that "Newton-step attenuation suppresses gSum error to <4e-8" is taken at face value and should be re-verified at iter=50 if the L3/L4 fix does not close the gap.
- All citations are at HEAD `b0c853a6f6` on branch `mlx/sprint-30-cosine-kahan`; line numbers may shift slightly under upstream merges.
