#pragma once
#include "full_features.h"

#include <catboost/libs/data/pair.h>

class TTrainData {
public:
    int LearnSampleCount;
    TAllFeatures AllFeatures;
    yvector<yvector<double>> Baseline;
    yvector<float> Target;
    yvector<float> Weights;
    yvector<ui32> QueryId;
    yhash<ui32, ui32> QuerySize;
    yvector<TPair> Pairs;

    ssize_t GetSampleCount() const {
        return Target.ysize();
    }
};
