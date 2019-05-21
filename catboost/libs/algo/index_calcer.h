#pragma once

#include "features_data_helpers.h"

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/options/restrictions.h>

#include <util/generic/vector.h>


class TFold;
struct TFullModel;
struct TSplit;
struct TSplitTree;

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
    const TSplitTree& tree,
    NCB::TTrainingForCPUDataProviderPtr learnData, // can be nullptr
    TConstArrayRef<NCB::TTrainingForCPUDataProviderPtr> testData, // can be empty
    NPar::TLocalExecutor* localExecutor);

TVector<ui8> GetModelCompatibleQuantizedFeatures(
    const TFullModel& model,
    const NCB::TObjectsDataProvider& objectsData,
    size_t start,
    size_t end);

TVector<ui8> GetModelCompatibleQuantizedFeatures(
    const TFullModel& model,
    const NCB::TObjectsDataProvider& objectsData);

TVector<TIndexType> BuildIndicesForBinTree(
    const TFullModel& model,
    const TVector<ui8>& binarizedFeatures,
    size_t treeId);


