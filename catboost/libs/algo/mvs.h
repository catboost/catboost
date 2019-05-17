#pragma once

#include <catboost/libs/options/enums.h>

#include <util/system/types.h>


class TFold;
struct TRestorableFastRng64;

namespace NPar {
    class TLocalExecutor;
}


class TMvsSampler {
public:
    TMvsSampler(ui32 sampleCount, float headFraction)
        : SampleCount(sampleCount)
        , HeadFraction(headFraction)
    {}
    float GetHeadFraction() const {
        return HeadFraction;
    }
    void GenSampleWeights(
        EBoostingType boostingType,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor,
        TFold* fold) const;
private:
    ui32 SampleCount;
    float HeadFraction;
};
