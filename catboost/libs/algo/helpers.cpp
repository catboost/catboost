#include "helpers.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <util/generic/algorithm.h>
#include <util/system/mem_info.h>

void GenerateBorders(const TPool& pool, TLearnContext* ctx, TVector<TFloatFeature>* floatFeatures) {
    auto& docStorage = pool.Docs;
    const THashSet<int>& categFeatures = ctx->CatFeatures;
    const auto& floatFeatureBorderOptions = ctx->Params.DataProcessingOptions->FloatFeaturesBinarization.Get();
    const int borderCount = floatFeatureBorderOptions.BorderCount;
    const ENanMode nanMode = floatFeatureBorderOptions.NanMode;
    const EBorderSelectionType borderType = floatFeatureBorderOptions.BorderSelectionType;

    size_t reasonCount = docStorage.GetFactorsCount() - categFeatures.size();
    floatFeatures->resize(reasonCount);
    if (reasonCount == 0) {
        return;
    }
    {
        size_t floatFeatureId = 0;
        for (int i = 0; i < docStorage.GetFactorsCount(); ++i) {
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
    // Estimate how many threads can generate borders
    const size_t bytes1M = 1024 * 1024, bytesThreadStack = 2 * bytes1M;
    const size_t bytesUsed = NMemInfo::GetMemInfo().RSS;
    const size_t bytesBestSplit = CalcMemoryForFindBestSplit(borderCount, docStorage.GetDocCount(), borderType);
    const size_t bytesGenerateBorders = sizeof(float) * docStorage.GetDocCount();
    const size_t bytesRequiredPerThread = bytesThreadStack + bytesGenerateBorders + bytesBestSplit;
    const auto usedRamLimit = ctx->Params.SystemOptions->CpuUsedRamLimit;
    const size_t threadCount = Min(reasonCount, (usedRamLimit - bytesUsed) / bytesRequiredPerThread);
    if (!(usedRamLimit >= bytesUsed && threadCount > 0)) {
        MATRIXNET_WARNING_LOG << "CatBoost needs " << (bytesUsed + bytesRequiredPerThread) / bytes1M + 1 << " Mb of memory to generate borders" << Endl;
    }

    TAtomic taskFailedBecauseOfNans = 0;
    THashSet<int> ignoredFeatureIndexes(ctx->Params.DataProcessingOptions->IgnoredFeatures->begin(), ctx->Params.DataProcessingOptions->IgnoredFeatures->end());
    auto calcOneFeatureBorder = [&](int idx) {
        auto& floatFeature = floatFeatures->at(idx);
        const auto floatFeatureIdx = floatFeatures->at(idx).FlatFeatureIndex;
        if (ignoredFeatureIndexes.has(floatFeatureIdx)) {
            return;
        }

        TVector<float> vals;

        floatFeature.HasNans = false;
        for (size_t i = 0; i < docStorage.GetDocCount(); ++i) {
            if (!IsNan(docStorage.Factors[floatFeatureIdx][i])) {
                vals.push_back(docStorage.Factors[floatFeatureIdx][i]);
            } else {
                floatFeature.HasNans = true;
            }
        }
        Sort(vals.begin(), vals.end());

        THashSet<float> borderSet = BestSplit(vals, borderCount, borderType);
        TVector<float> bordersBlock(borderSet.begin(), borderSet.end());
        Sort(bordersBlock.begin(), bordersBlock.end());
        if (floatFeature.HasNans) {
            if (nanMode == ENanMode::Min) {
                bordersBlock.insert(bordersBlock.begin(), std::numeric_limits<float>::lowest());
            } else if (nanMode == ENanMode::Max) {
                bordersBlock.push_back(std::numeric_limits<float>::max());
            } else {
                Y_ASSERT(nanMode == ENanMode::Forbidden);
                taskFailedBecauseOfNans = 1;
            }
        }
        floatFeature.Borders.swap(bordersBlock);
    };
    size_t nReason = 0;
    for (; nReason + threadCount <= reasonCount; nReason += threadCount) {
        ctx->LocalExecutor.ExecRange(calcOneFeatureBorder, nReason, nReason + threadCount, NPar::TLocalExecutor::WAIT_COMPLETE);
        CB_ENSURE(taskFailedBecauseOfNans == 0, "There are nan factors and nan values for float features are not allowed. Set nan_mode != Forbidden.");
    }
    for (; nReason < reasonCount; ++nReason) {
        calcOneFeatureBorder(nReason);
    }

    MATRIXNET_INFO_LOG << "Borders for float features generated" << Endl;
}

void ApplyPermutation(const TVector<size_t>& permutation, TPool* pool) {
    Y_VERIFY(pool->Docs.GetDocCount() == 0 || permutation.size() == pool->Docs.GetDocCount());

    TVector<size_t> toIndices(permutation);
    for (size_t i = 0; i < pool->Docs.GetDocCount(); ++i) {
        while (toIndices[i] != i) {
            auto destinationIndex = toIndices[i];
            pool->Docs.SwapDoc(i, destinationIndex);
            DoSwap(toIndices[i], toIndices[destinationIndex]);
        }
    }

    for (auto& pair : pool->Pairs) {
        pair.WinnerId = permutation[pair.WinnerId];
        pair.LoserId = permutation[pair.LoserId];
    }
}

TVector<size_t> InvertPermutation(const TVector<size_t>& permutation) {
    TVector<size_t> result(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        result[permutation[i]] = i;
    }
    return result;
}

int GetClassesCount(const TVector<float>& target, int classesCount) {
    int maxClass = static_cast<int>(*MaxElement(target.begin(), target.end()));
    if (classesCount == 0) { // classesCount not set
        return maxClass + 1;
    } else {
        CB_ENSURE(maxClass < classesCount, "if classes-count is specified then each target label should be in range 0,..,classes_count-1");
        return classesCount;
    }
}
