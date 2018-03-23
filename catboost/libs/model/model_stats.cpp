#include "model_stats.h"

#include <catboost/libs/options/restrictions.h>
#include <catboost/libs/algo/index_calcer.h>

TVector<TVector<float>> ComputeTotalLeafWeights(const TPool& pool, const TFullModel& model) {
    const size_t treeCount = model.ObliviousTrees.TreeSizes.size();
    TVector<TVector<float>> leafWeights(treeCount);
    for (size_t index = 0; index < treeCount; ++index) {
        leafWeights[index].resize(model.ObliviousTrees.LeafValues[index].size() / model.ObliviousTrees.ApproxDimension);
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
        for (size_t doc = 0; doc < documentsCount; ++doc) {
            const TIndexType valueIndex = indices[doc];
            leafWeights[treeIdx][valueIndex] += pool.Docs.Weight[doc];
        }
    }
    return leafWeights;
}
