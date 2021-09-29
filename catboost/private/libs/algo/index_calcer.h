#pragma once

#include "model_quantization_adapter.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/objects.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/hash.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


class TFold;
struct TSplit;
struct TSplitTree;
struct TNonSymmetricTreeStructure;

namespace NPar {
    class ILocalExecutor;
}


/*
 * for index calculation we need either TFullSubset or TIndexedSubset
 * so, for TBlockedSubset we need to make it TIndexedSubset
 */
using TIndexedSubsetCache = THashMap<const NCB::TFeaturesArraySubsetIndexing*, NCB::TIndexedSubset<ui32>>;


void GetObjectsDataAndIndexing(
    const NCB::TTrainingDataProviders& trainingData,
    const TFold& fold,
    bool isEstimated,
    bool isOnline,
    ui32 objectSubsetIdx, // 0 - learn, 1+ - test (subtract 1 for testIndex)
    TIndexedSubsetCache* indexedSubsetCache,
    NPar::ILocalExecutor* localExecutor,
    NCB::TQuantizedObjectsDataProviderPtr* objectsData,
    const ui32** columnIndexing // can return nullptr
);

void SetPermutedIndices(
    const TSplit& split,
    const NCB::TTrainingDataProviders& trainingData,
    int curDepth,
    const TFold& fold,
    TArrayRef<TIndexType> indices,
    NPar::ILocalExecutor* localExecutor);

TVector<bool> GetIsLeafEmpty(int curDepth, TConstArrayRef<TIndexType> indices, NPar::ILocalExecutor* localExecutor);

int GetRedundantSplitIdx(const TVector<bool>& isLeafEmpty);


enum class EBuildIndicesDataParts {
    All,
    LearnOnly,
    TestOnly
};


TVector<TIndexType> BuildIndices(
    const TFold& fold, // can be empty
    const std::variant<TSplitTree, TNonSymmetricTreeStructure>& tree,
    const NCB::TTrainingDataProviders& trainingData,
    EBuildIndicesDataParts dataParts,
    NPar::ILocalExecutor* localExecutor);

TVector<TIndexType> BuildIndicesForBinTree(
    const TFullModel& model,
    const NCB::NModelEvaluation::IQuantizedData* quantizedFeatures,
    size_t treeId);


