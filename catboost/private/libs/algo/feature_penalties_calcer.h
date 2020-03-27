#pragma once

#include "tensor_search_helpers.h"

#include <catboost/private/libs/options/feature_penalties_options.h>

namespace NCB {
    void AddFeaturePenaltiesToBestSplits(
        TLearnContext* ctx,
        const NCB::TQuantizedForCPUObjectsDataProvider& objectsData,
        ui32 oneHotMaxSize,
        TVector<TCandidateInfo>* candidates
    );
}