#include "helpers.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <util/generic/algorithm.h>
#include <util/system/mem_info.h>

void GenerateBorders(const TDocumentStorage& docStorage, TLearnContext* ctx, yvector<yvector<float>>* borders, yvector<bool>* hasNans) {

    const yhash_set<int>& categFeatures = ctx->CatFeatures;
    const int borderCount = ctx->Params.BorderCount;
    const EBorderSelectionType borderType = ctx->Params.FeatureBorderType;

    size_t reasonCount = docStorage.GetFactorsCount() - categFeatures.size();
    borders->resize(reasonCount);
    hasNans->resize(reasonCount);
    if (reasonCount == 0) {
        return;
    }

    yvector<int> floatIndexes;
    for (int i = 0; i < docStorage.GetFactorsCount(); ++i) {
        if (!categFeatures.has(i)) {
            floatIndexes.push_back(i);
        }
    }
    // Estimate how many threads can generate borders
    const size_t bytes1M = 1024 * 1024, bytesThreadStack = 2 * bytes1M;
    const size_t bytesUsed = NMemInfo::GetMemInfo().RSS;
    const size_t bytesBestSplit = (sizeof(float) + (borderCount - 1) * sizeof(size_t) + 2 * sizeof(double) + 2 * sizeof(size_t) + 2 * sizeof(double)) * docStorage.GetDocCount();
    const size_t bytesGenerateBorders = sizeof(float) * docStorage.GetDocCount();
    const size_t bytesRequiredPerThread = bytesThreadStack + bytesGenerateBorders + bytesBestSplit;
    const size_t threadCount = Min(reasonCount, (ctx->Params.UsedRAMLimit - bytesUsed) / bytesRequiredPerThread);
    if (!(ctx->Params.UsedRAMLimit >= bytesUsed && threadCount > 0)) {
        MATRIXNET_WARNING_LOG << "CatBoost needs " << (bytesUsed + bytesRequiredPerThread) / bytes1M + 1 << " Mb of memory to generate borders" << Endl;
    }

    TAtomic taskFailedBecauseOfNans = 0;
    yhash_set<int> ignoredFeatureIndexes(ctx->Params.IgnoredFeatures.begin(), ctx->Params.IgnoredFeatures.end());
    auto calcOneFeatureBorder = [&](int idx) {
        const auto floatFeatureIdx = floatIndexes[idx];
        if (ignoredFeatureIndexes.has(floatFeatureIdx)) {
            return;
        }

        yvector<float> vals;

        (*hasNans)[idx] = false;
        for (size_t i = 0; i < docStorage.GetDocCount(); ++i) {
            if (!IsNan(docStorage.Factors[floatFeatureIdx][i])) {
                vals.push_back(docStorage.Factors[floatFeatureIdx][i]);
            } else {
                (*hasNans)[idx] = true;
            }
        }
        Sort(vals.begin(), vals.end());

        yhash_set<float> borderSet = BestSplit(vals, borderCount, borderType);
        yvector<float> bordersBlock(borderSet.begin(), borderSet.end());
        Sort(bordersBlock.begin(), bordersBlock.end());
        if ((*hasNans)[idx]) {
            if (ctx->Params.NanMode == ENanMode::Min) {
                bordersBlock.insert(bordersBlock.begin(), std::numeric_limits<float>::lowest());
            } else if (ctx->Params.NanMode == ENanMode::Max) {
                bordersBlock.push_back(std::numeric_limits<float>::max());
            } else {
                Y_ASSERT(ctx->Params.NanMode == ENanMode::Forbidden);
                taskFailedBecauseOfNans = 1;
            }
        }
        (*borders)[idx].swap(bordersBlock);
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

void ApplyPermutation(const yvector<size_t>& permutation, TPool* pool) {
    Y_VERIFY(pool->Docs.GetDocCount() == 0 || permutation.size() == pool->Docs.GetDocCount());

    yvector<size_t> toIndices(permutation);
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

yvector<size_t> InvertPermutation(const yvector<size_t>& permutation) {
    yvector<size_t> result(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        result[permutation[i]] = i;
    }
    return result;
}

int GetClassesCount(const yvector<float>& target, int classesCount) {
    int maxClass = static_cast<int>(*MaxElement(target.begin(), target.end()));
    if (classesCount == 0) { // classesCount not set
        return maxClass + 1;
    } else {
        CB_ENSURE(maxClass < classesCount, "if classes-count is specified then each target label should be in range 0,..,classes_count-1");
        return classesCount;
    }
}
