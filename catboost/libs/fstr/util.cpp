#include "util.h"

#include <catboost/libs/algo/index_calcer.h>


TVector<TVector<double>> CollectLeavesStatistics(const TPool& pool, const TFullModel& model) {
    const size_t treeCount = model.ObliviousTrees.TreeSizes.size();
    TVector<TVector<double>> leavesStatistics(treeCount);
    for (size_t index = 0; index < treeCount; ++index) {
        leavesStatistics[index].resize(1 << model.ObliviousTrees.TreeSizes[index]);
    }

    auto binFeatures = BinarizeFeatures(model, pool);

    const auto documentsCount = pool.Docs.GetDocCount();
    for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        TVector<TIndexType> indices = BuildIndicesForBinTree(
            model,
            binFeatures,
            treeIdx);

        if (indices.empty()) {
            continue;
        }

        if (pool.Docs.Weight.empty()) {
            for (size_t doc = 0; doc < documentsCount; ++doc) {
                const TIndexType valueIndex = indices[doc];
                leavesStatistics[treeIdx][valueIndex] += 1.0;
            }
        } else {
            for (size_t doc = 0; doc < documentsCount; ++doc) {
                const TIndexType valueIndex = indices[doc];
                leavesStatistics[treeIdx][valueIndex] += pool.Docs.Weight[doc];
            }
        }
    }
    return leavesStatistics;
}

