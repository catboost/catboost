#pragma once

#include "fold.h"
#include "learn_context.h"
#include "split.h"

#include <catboost/libs/model/model.h>
#include <catboost/libs/model/formula_evaluator.h>

#include <util/generic/vector.h>

void SetPermutedIndices(const TSplit& split,
                        const TAllFeatures& features,
                        int curDepth,
                        const TFold& fold,
                        TVector<TIndexType>* indices,
                        TLearnContext* ctx);

int GetRedundantSplitIdx(int curDepth, const TVector<TIndexType>& indices);

void DeleteSplit(int curDepth, int redundantIdx, TSplitTree* tree, TVector<TIndexType>* indices);

TVector<TIndexType> BuildIndices(const TFold& fold,
                          const TSplitTree& tree,
                          const TTrainData& data,
                          NPar::TLocalExecutor* localExecutor);

int GetDocCount(const TAllFeatures& features);

inline TVector<ui8> BinarizeFeatures(const TFullModel& model, const TPool& pool) {
    auto docCount = pool.Docs.GetDocCount();
    TVector<ui8> result(model.ObliviousTrees.GetBinaryFeaturesCount() * docCount);
    TVector<int> transposedHash(docCount * model.ObliviousTrees.CatFeatures.size());
    TVector<float> ctrs(model.ObliviousTrees.GetUsedModelCtrs().size() * docCount);
    BinarizeFeatures(model,
                     [&](const TFloatFeature& floatFeature, size_t index) {
                         return pool.Docs.Factors[floatFeature.FlatFeatureIndex][index];
                     },
                     [&](size_t catFeatureIdx, size_t index) {
                         return ConvertFloatCatFeatureToIntHash(pool.Docs.Factors[model.ObliviousTrees.CatFeatures[catFeatureIdx].FlatFeatureIndex][index]);
                     },
                     0,
                     docCount,
                     result,
                     transposedHash,
                     ctrs);
    return result;
}

inline TVector<TIndexType> BuildIndicesForBinTree(const TFullModel& model,
                                           const TVector<ui8>& binarizedFeatures,
                                           size_t treeId) {
    auto docCount = binarizedFeatures.size() / model.ObliviousTrees.GetBinaryFeaturesCount();
    TVector<TIndexType> indexesVec(docCount);
    const int* treeSplitsCurPtr =
        model.ObliviousTrees.TreeSplits.data() +
        model.ObliviousTrees.TreeStartOffsets[treeId];
    auto curTreeSize = model.ObliviousTrees.TreeSizes[treeId];
    for (int depth = 0; depth < curTreeSize; ++depth) {
        auto indexesPtr = indexesVec.data();
        const auto bin = treeSplitsCurPtr[depth];
        auto binFeaturePtr = &binarizedFeatures[bin * docCount];
        for (size_t docId = 0; docId < docCount; ++docId) {
            indexesPtr[docId] |= binFeaturePtr[docId] << depth;
        }
    }
    return indexesVec;
}
