#pragma once

#include "full_features.h"

#include <catboost/libs/data/pair.h>

class TTrainData {
public:
    int LearnSampleCount;
    TAllFeatures AllFeatures;
    TVector<TVector<double>> Baseline;
    TVector<float> Target;
    TVector<float> Weights;
    TVector<ui32> QueryId;
    THashMap<ui32, ui32> QuerySize;
    TVector<TPair> Pairs;

    ssize_t GetSampleCount() const {
        return Target.ysize();
    }
};
