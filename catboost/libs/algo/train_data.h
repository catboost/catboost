#pragma once

#include "full_features.h"

#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/data_types/query.h>

class TTrainData {
public:
    int LearnSampleCount = 0;
    int LearnQueryCount = 0;
    int LearnPairsCount = 0;
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

template <typename T> T Concat(const T& a, const T& b);
template <typename T> TVector<T> Concat(const TVector<T>& a, const TVector<T>& b);
