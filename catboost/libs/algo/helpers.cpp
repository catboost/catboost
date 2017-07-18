#include "helpers.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <util/generic/algorithm.h>

yvector<yvector<float>> GenerateBorders(
    const yvector<TDocInfo>& docInfos,
    const yhash_set<int>& categFeatures,
    NPar::TLocalExecutor& localExecutor,
    int borderCount,
    const EBorderSelectionType borderType)
{
    int reasonCount = docInfos[0].Factors.size();
    yvector<yvector<float>> res(reasonCount);

    auto calcOneFeatureBorder = [&](int nReason) {
        if (categFeatures.has(nReason)) {
            return;
        }

        yvector<float> vals;
        for (int i = 0; i < docInfos.ysize(); ++i) {
            vals.push_back(docInfos[i].Factors[nReason]);
        }
        Sort(vals.begin(), vals.end());

        yhash_set<float> borderSet = BestSplit(vals, borderCount, borderType);
        yvector<float> borders(borderSet.begin(), borderSet.end());
        Sort(borders.begin(), borders.end());
        res[nReason].swap(borders);
    };
    localExecutor.ExecRange(calcOneFeatureBorder, 0, reasonCount, NPar::TLocalExecutor::WAIT_COMPLETE);

    MATRIXNET_INFO_LOG << "Borders generated" << Endl;
    return res;
}

void ApplyPermutation(const yvector<size_t>& permutation, TPool* pool) {
    Y_VERIFY(permutation.ysize() == pool->Docs.ysize());

    yvector<size_t> toIndices(permutation);
    for (size_t i = 0; i < permutation.size(); ++i) {
        while (toIndices[i] != i) {
            auto destinationIndex = toIndices[i];
            DoSwap(pool->Docs[i], pool->Docs[destinationIndex]);
            DoSwap(toIndices[i], toIndices[destinationIndex]);
        }
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
