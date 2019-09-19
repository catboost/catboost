#pragma once

#include <catboost/libs/options/enums.h>

#include <util/system/types.h>
#include <util/generic/vector.h>


class TFold;
struct TRestorableFastRng64;

namespace NPar {
    class TLocalExecutor;
}


class TMvsSampler {
public:
    TMvsSampler(ui32 sampleCount, float sampleRate, float lambda, bool lambdaIsSet)
        : SampleCount(sampleCount)
        , SampleRate(sampleRate)
        , Lambda(lambda)
        , LambdaIsSet(lambdaIsSet)
    {}
    float GetSampleRate() const {
        return SampleRate;
    }
    double GetLambda(
        const double* derivatives,
        const TVector<TVector<TVector<double>>>& leafValues,
        NPar::TLocalExecutor* localExecutor) const;
    double CalculateThreshold(
        TVector<double>::iterator candidatesBegin,
        TVector<double>::iterator candidatesEnd,
        double sumSmall,
        ui32 nLarge,
        double sampleSize) const;
    void GenSampleWeights(
        EBoostingType boostingType,
        const TVector<TVector<TVector<double>>>& leafValues,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor,
        TFold* fold) const;
private:
    ui32 SampleCount;
    float SampleRate;
    float Lambda;
    bool LambdaIsSet;
};
