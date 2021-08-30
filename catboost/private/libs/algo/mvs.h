#pragma once

#include <catboost/private/libs/options/enums.h>

#include <util/system/types.h>
#include <util/generic/vector.h>
#include <util/generic/maybe.h>
#include <util/generic/array_ref.h>


class TFold;
struct TRestorableFastRng64;

namespace NPar {
    class ILocalExecutor;
}


class TMvsSampler {
public:
    TMvsSampler(ui32 sampleCount, float sampleRate, const TMaybe<float>& lambda)
        : SampleCount(sampleCount)
        , SampleRate(sampleRate)
        , Lambda(lambda)
    {}
    void GenSampleWeights(
        EBoostingType boostingType,
        const TVector<TVector<TVector<double>>>& leafValues,
        TRestorableFastRng64* rand,
        NPar::ILocalExecutor* localExecutor,
        TFold* fold) const;

private:
    double GetLambda(
        const TVector<TConstArrayRef<double>>& derivatives,
        const TVector<TVector<TVector<double>>>& leafValues,
        NPar::ILocalExecutor* localExecutor) const;
    double CalculateThreshold(
        TVector<double>::iterator candidatesBegin,
        TVector<double>::iterator candidatesEnd,
        double sumSmall,
        ui32 nLarge,
        double sampleSize) const;

private:
    ui32 SampleCount;
    float SampleRate;
    const ui32 BlockSize = 8192;
    TMaybe<float> Lambda;
};
