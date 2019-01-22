#pragma once

#include "fold.h"
#include "split.h"

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/options/restrictions.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>


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

struct TFullModel;

void BinarizeFeatures(
    const TFullModel& model,
    const NCB::TRawObjectsDataProvider& rawObjectsData,
    size_t start,
    size_t end,
    TVector<ui8>* result);

TVector<ui8> BinarizeFeatures(
    const TFullModel& model,
    const NCB::TRawObjectsDataProvider& rawObjectsData,
    size_t start,
    size_t end);

TVector<ui8> BinarizeFeatures(const TFullModel& model, const NCB::TRawObjectsDataProvider& rawObjectsData);

TVector<TIndexType> BuildIndicesForBinTree(
    const TFullModel& model,
    const TVector<ui8>& binarizedFeatures,
    size_t treeId);
