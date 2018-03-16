#pragma once

#include "full_features.h"

#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/data_types/query.h>

class TTrainData {
public:
    TAllFeatures AllFeatures;
    TVector<TVector<double>> Baseline;
    TVector<float> Target;
    TVector<float> Weights;
    TVector<ui32> QueryId;
    TVector<ui32> SubgroupId;
    TVector<TQueryInfo> QueryInfo;
    TVector<TPair> Pairs;

    ssize_t GetSampleCount() const {
        return Target.ysize();
    }

    ssize_t GetQueryCount() const {
        return QueryInfo.ysize();
    }
};
