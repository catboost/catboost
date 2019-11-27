#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/fwd.h>
#include <util/generic/vector.h>


class TFold;
struct TSplit;
struct TNonSymmetricTreeStructure;
struct TOnlineCTR;

namespace NCB {
    class TObjectsDataProvider;
    class TQuantizedForCPUObjectsDataProvider;
}

namespace NPar {
    class TLocalExecutor;
}

void BuildIndicesForDataset(
    const TNonSymmetricTreeStructure& tree,
    const NCB::TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const NCB::TFeaturesArraySubsetIndexing& featuresArraySubsetIndexing,
    ui32 sampleCount,
    const TVector<const TOnlineCTR*>& onlineCtrs,
    ui32 docOffset,
    NPar::TLocalExecutor* localExecutor,
    TIndexType* indices);
