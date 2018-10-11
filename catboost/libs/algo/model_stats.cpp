#include "model_stats.h"

#include "index_calcer.h"

#include <catboost/libs/options/restrictions.h>


TVector<TVector<double>> ComputeTotalLeafWeights(const TPool& pool, const TFullModel& model) {
    const size_t treeCount = model.ObliviousTrees.TreeSizes.size();
    TVector<TVector<double>> leafWeights(treeCount);
    for (size_t index = 0; index < treeCount; ++index) {
        leafWeights[index].resize((1uLL << model.ObliviousTrees.TreeSizes[index]));
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
        if (pool.MetaInfo.HasWeights) {
            for (size_t doc = 0; doc < documentsCount; ++doc) {
                const TIndexType valueIndex = indices[doc];
                leafWeights[treeIdx][valueIndex] += pool.Docs.Weight[doc];
            }
        } else {
            for (size_t doc = 0; doc < documentsCount; ++doc) {
                const TIndexType valueIndex = indices[doc];
                leafWeights[treeIdx][valueIndex] += 1.0;
            }
        }
    }
    return leafWeights;
}
