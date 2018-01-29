#pragma once

#include "full_features.h"

#include <catboost/libs/data/pair.h>
#include <catboost/libs/data/query.h>

class TTrainData {
public:
    int LearnSampleCount;
    int LearnQueryCount;
    TAllFeatures AllFeatures;
    TVector<TVector<double>> Baseline;
    TVector<float> Target;
    TVector<float> Weights;
    TVector<ui32> QueryId;
    TVector<TQueryInfo> QueryInfo;
    TVector<TPair> Pairs;

    ssize_t GetSampleCount() const {
        return Target.ysize();
    }

    ssize_t GetQueryCount() const {
        return QueryInfo.ysize();
    }
};
