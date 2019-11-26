#pragma once

#include "model_quantization_adapter.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/vector.h>


class TFold;
struct TSplit;
struct TSplitTree;
struct TNonSymmetricTreeStructure;

namespace NCB {
    class TObjectsDataProvider;
    class TQuantizedForCPUObjectsDataProvider;
}

namespace NPar {
    class TLocalExecutor;
}


void SetPermutedIndices(
    const TSplit& split,
    const NCB::TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    int curDepth,
    const TFold& fold,
    TVector<TIndexType>* indices,
    NPar::TLocalExecutor* localExecutor);

TVector<bool> GetIsLeafEmpty(int curDepth, const TVector<TIndexType>& indices);

int GetRedundantSplitIdx(const TVector<bool>& isLeafEmpty);

TVector<TIndexType> BuildIndices(
    const TFold& fold, // can be empty
    const TVariant<TSplitTree, TNonSymmetricTreeStructure>& tree,
    NCB::TTrainingForCPUDataProviderPtr learnData, // can be nullptr
    TConstArrayRef<NCB::TTrainingForCPUDataProviderPtr> testData, // can be empty
    NPar::TLocalExecutor* localExecutor);

TVector<TIndexType> BuildIndicesForBinTree(
    const TFullModel& model,
    const NCB::NModelEvaluation::IQuantizedData* quantizedFeatures,
    size_t treeId);


