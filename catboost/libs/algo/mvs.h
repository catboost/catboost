#pragma once
#include "fold.h"
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/options/enums.h>
#include <library/threading/local_executor/fwd.h>


class TMvsSampler {
public:
    TMvsSampler(ui32 sampleCount, float headFraction)
        : SampleCount(sampleCount)
        , HeadFraction(headFraction)
    {}
    float GetHeadFraction() const {
        return HeadFraction;
    }
    void GenSampleWeights(TFold& fold, EBoostingType boostingType, TRestorableFastRng64* rand, NPar::TLocalExecutor* localExecutor) const;
private:
    ui32 SampleCount;
    float HeadFraction;
};
