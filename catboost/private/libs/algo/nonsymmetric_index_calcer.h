#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/fwd.h>
#include <util/generic/vector.h>


class TFold;
struct TSplitNode;
struct TNonSymmetricTreeStructure;
class TOnlineCtrBase;

namespace NPar {
    class ILocalExecutor;
}


void UpdateIndicesWithSplit(
    const TSplitNode& node,
    const NCB::TTrainingDataProviders& trainingData,
    const NCB::TIndexedSubset<ui32>& docsSubset,
    const TFold& fold,
    NPar::ILocalExecutor* localExecutor,
    TArrayRef<TIndexType> indices,
    NCB::TIndexedSubset<ui32>* leftIndices,
    NCB::TIndexedSubset<ui32>* rightIndices
);

void UpdateIndices(
    const TSplitNode& node,
    const NCB::TTrainingDataProviders& trainingData,
    const NCB::TIndexedSubset<ui32>& docsSubset,
    const TFold& fold,
    NPar::ILocalExecutor* localExecutor,
    TArrayRef<TIndexType> indices
);

void BuildIndicesForDataset(
    const TNonSymmetricTreeStructure& tree,
    const NCB::TTrainingDataProviders& trainingData,
    const TFold& fold,
    ui32 sampleCount,
    const TVector<const TOnlineCtrBase*>& onlineCtrs,
    ui32 objectSubsetIdx, // 0 - learn, 1+ - test (subtract 1 for testIndex)
    NPar::ILocalExecutor* localExecutor,
    TIndexType* indices);
