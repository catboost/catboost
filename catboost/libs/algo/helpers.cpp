#include "helpers.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <library/malloc/api/malloc.h>

#include <util/generic/algorithm.h>
#include <util/generic/utility.h>
#include <util/system/mem_info.h>

void GenerateBorders(const TPool& pool, TLearnContext* ctx, TVector<TFloatFeature>* floatFeatures) {
    auto& docStorage = pool.Docs;
    const THashSet<int>& categFeatures = ctx->CatFeatures;
    const auto& floatFeatureBorderOptions = ctx->Params.DataProcessingOptions->FloatFeaturesBinarization.Get();
    const int borderCount = floatFeatureBorderOptions.BorderCount;
    const ENanMode nanMode = floatFeatureBorderOptions.NanMode;
    const EBorderSelectionType borderType = floatFeatureBorderOptions.BorderSelectionType;

    size_t reasonCount = docStorage.GetEffectiveFactorCount() - categFeatures.size();
    floatFeatures->resize(reasonCount);
    if (reasonCount == 0) {
        return;
    }
    {
        size_t floatFeatureId = 0;
        for (int i = 0; i < docStorage.GetEffectiveFactorCount(); ++i) {
            if (categFeatures.has(i)) {
                continue;
            }
            auto& floatFeature = floatFeatures->at(floatFeatureId);
            floatFeature.FeatureIndex = static_cast<int>(floatFeatureId);
            floatFeature.FlatFeatureIndex = i;
            if (i < pool.FeatureId.ysize()) {
                floatFeature.FeatureId = pool.FeatureId[i];
            }
            ++floatFeatureId;
        }
    }
    size_t samplesToBuildBorders = docStorage.GetDocCount();
    bool isSubsampled = false;
    const constexpr size_t SlowSubsampleSize = 200 * 1000;
    // Use random 200K documents to build borders for slow MinEntropy and MaxLogSum
    // Create random shuffle if HasTimeFlag
    if (EqualToOneOf(borderType, EBorderSelectionType::MinEntropy, EBorderSelectionType::MaxLogSum) && samplesToBuildBorders > SlowSubsampleSize) {
        samplesToBuildBorders = SlowSubsampleSize;
        isSubsampled = true;
    }
    TVector<size_t> randomShuffle;
    const bool isShuffleNeeded = isSubsampled && ctx->Params.DataProcessingOptions->HasTimeFlag;
    if (isShuffleNeeded) {
        randomShuffle.yresize(docStorage.GetDocCount());
        std::iota(randomShuffle.begin(), randomShuffle.end(), 0);
        Shuffle(randomShuffle.begin(), randomShuffle.end(), ctx->Rand);
    }
    // Estimate how many threads can generate borders
    const size_t bytes1M = 1024 * 1024, bytesThreadStack = 2 * bytes1M;
    const size_t bytesUsed = NMemInfo::GetMemInfo().RSS;
    const size_t bytesBestSplit = CalcMemoryForFindBestSplit(borderCount, samplesToBuildBorders, borderType);
    const size_t bytesGenerateBorders = sizeof(float) * samplesToBuildBorders;
    const size_t bytesRequiredPerThread = bytesThreadStack + bytesGenerateBorders + bytesBestSplit;
    const size_t usedRamLimit = ParseMemorySizeDescription(ctx->Params.SystemOptions->CpuUsedRamLimit);
    const i64 availableMemory = (i64)usedRamLimit - bytesUsed;
    const size_t threadCount = availableMemory > 0 ? Min(reasonCount, (ui64)availableMemory / bytesRequiredPerThread) : 1;
    if (!(usedRamLimit >= bytesUsed)) {
        MATRIXNET_WARNING_LOG << "CatBoost needs " << (bytesUsed + bytesRequiredPerThread) / bytes1M + 1 << " Mb of memory to generate borders" << Endl;
    }
    TAtomic taskFailedBecauseOfNans = 0;
    THashSet<int> ignoredFeatureIndexes(ctx->Params.DataProcessingOptions->IgnoredFeatures->begin(), ctx->Params.DataProcessingOptions->IgnoredFeatures->end());
    auto calcOneFeatureBorder = [&](int idx) {
        auto& floatFeature = floatFeatures->at(idx);
        const auto floatFeatureIdx = floatFeature.FlatFeatureIndex;
        if (ignoredFeatureIndexes.has(floatFeatureIdx)) {
            return;
        }

        TVector<float> vals;
        vals.reserve(samplesToBuildBorders);
        for (size_t i = 0; i < samplesToBuildBorders; ++i) {
            const size_t randomDocIdx = isShuffleNeeded ? randomShuffle[i] : i;
            const float factor = docStorage.Factors[floatFeatureIdx][randomDocIdx];
            if (!IsNan(factor)) {
                vals.push_back(factor);
            }
        }

        THashSet<float> borderSet = BestSplit(vals, borderCount, borderType);
        if (borderSet.has(-0.0f)) { // BestSplit might add negative zeros
            borderSet.erase(-0.0f);
            borderSet.insert(0.0f);
        }
        TVector<float> bordersBlock(borderSet.begin(), borderSet.end());
        Sort(bordersBlock.begin(), bordersBlock.end());

        floatFeature.HasNans = AnyOf(docStorage.Factors[floatFeatureIdx], IsNan);
        if (floatFeature.HasNans) {
            if (nanMode == ENanMode::Min) {
                floatFeature.NanValueTreatment = NCatBoostFbs::ENanValueTreatment_AsFalse;
                bordersBlock.insert(bordersBlock.begin(), std::numeric_limits<float>::lowest());
            } else if (nanMode == ENanMode::Max) {
                floatFeature.NanValueTreatment = NCatBoostFbs::ENanValueTreatment_AsTrue;
                bordersBlock.push_back(std::numeric_limits<float>::max());
            } else {
                Y_ASSERT(nanMode == ENanMode::Forbidden);
                taskFailedBecauseOfNans = 1;
            }
        }
        floatFeature.Borders.swap(bordersBlock);
    };
    size_t nReason = 0;
    if (threadCount > 1) {
        for (; nReason + threadCount <= reasonCount; nReason += threadCount) {
            ctx->LocalExecutor.ExecRange(calcOneFeatureBorder, nReason, nReason + threadCount,
                                         NPar::TLocalExecutor::WAIT_COMPLETE);
            CB_ENSURE(taskFailedBecauseOfNans == 0,
                      "There are nan factors and nan values for float features are not allowed. Set nan_mode != Forbidden.");
        }
    }
    for (; nReason < reasonCount; ++nReason) {
        calcOneFeatureBorder(nReason);
    }

    MATRIXNET_INFO_LOG << "Borders for float features generated" << Endl;
}

void ConfigureMalloc() {
#if !(defined(__APPLE__) && defined(__MACH__)) // there is no LF for MacOS
    if (!NMalloc::MallocInfo().SetParam("LB_LIMIT_TOTAL_SIZE", "1000000")) {
        MATRIXNET_DEBUG_LOG << "link with lfalloc for better performance" << Endl;
    }
#endif
}

void CalcErrors(
    const TDataset& learnData,
    const TDatasetPtrs& testDataPtrs,
    const TVector<THolder<IMetric>>& errors,
    bool calcMetrics,
    TLearnContext* ctx
) {
    if (learnData.GetSampleCount() > 0) {
        TVector<bool> skipMetricOnTrain = GetSkipMetricOnTrain(errors);
        const auto& data = learnData;
        ctx->LearnProgress.MetricsAndTimeHistory.LearnMetricsHistory.emplace_back();
        for (int i = 0; i < errors.ysize(); ++i) {
            const TMap<TString, TString> hints = errors[i]->GetHints();
            if (calcMetrics && !skipMetricOnTrain[i]) {
                ctx->LearnProgress.MetricsAndTimeHistory.LearnMetricsHistory.back().push_back(EvalErrors(
                    ctx->LearnProgress.AvrgApprox,
                    data.Target,
                    data.Weights,
                    data.QueryInfo,
                    errors[i],
                    &ctx->LocalExecutor
                ));
            }
        }
    }

    if (GetSampleCount(testDataPtrs) > 0) {
        ctx->LearnProgress.MetricsAndTimeHistory.TestMetricsHistory.emplace_back(); // new [iter]
        auto& testMetricErrors = ctx->LearnProgress.MetricsAndTimeHistory.TestMetricsHistory.back();
        for (size_t testIdx = 0; testIdx < testDataPtrs.size(); ++testIdx) {
            testMetricErrors.emplace_back();
            if (testDataPtrs[testIdx] == nullptr || testDataPtrs[testIdx]->GetSampleCount() == 0) {
                continue;
            }
            const auto& testApprox = ctx->LearnProgress.TestApprox[testIdx];
            const auto& data = *testDataPtrs[testIdx];
            for (int i = 0; i < errors.ysize(); ++i) {
                if (i == 0 || calcMetrics) { // TODO(smirnovpavel): Decide what to do with eval_metric if metric_period != 1. Decide what to do with custom objectives when no metric is present.
                    testMetricErrors.back().push_back(EvalErrors(
                        testApprox,
                        data.Target,
                        data.Weights,
                        data.QueryInfo,
                        errors[i],
                        &ctx->LocalExecutor
                    ));
                }
            }
        }
    }
}
