#pragma once

#include "full_features.h"

#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/data_types/query.h>

#include <library/binsaver/bin_saver.h>

class TDataset {
public:
    TAllFeatures AllFeatures;
    TVector<TVector<double>> Baseline;
    TVector<float> Target;
    TVector<float> Weights;
    TVector<TGroupId> QueryId;
    TVector<ui32> SubgroupId;
    TVector<TQueryInfo> QueryInfo;
    TVector<TPair> Pairs;

    ssize_t GetSampleCount() const {
        return Target.ysize();
    }

    ssize_t GetQueryCount() const {
        return QueryInfo.ysize();
    }
    SAVELOAD(AllFeatures, Baseline, Target, Weights, QueryId, SubgroupId, QueryInfo, Pairs);
};
