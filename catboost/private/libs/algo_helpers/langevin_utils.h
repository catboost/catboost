#pragma once

#include <util/generic/fwd.h>
#include <util/system/types.h>
#include <cmath>

struct TSum;
struct TSumMulti;

namespace NPar {
    class ILocalExecutor;
}

double CalcLangevinNoiseRate(float diffusionTemperature, float learningRate);

void AddLangevinNoiseToDerivatives(
    float diffusionTemperature,
    float learningRate,
    ui64 randomSeed,
    TVector<double>* derivatives,
    NPar::ILocalExecutor* localExecutor
);

void AddLangevinNoiseToDerivatives(
    float diffusionTemperature,
    float learningRate,
    ui64 randomSeed,
    TVector<TVector<double>>* derivatives,
    NPar::ILocalExecutor* localExecutor
);

void AddLangevinNoiseToLeafDerivativesSum(
    float diffusionTemperature,
    float learningRate,
    double scaledL2Regularizer,
    ui64 randomSeed,
    TVector<TSum>* leafDersSum
);

void AddLangevinNoiseToLeafDerivativesSum(
    float diffusionTemperature,
    float learningRate,
    double scaledL2Regularizer,
    ui64 randomSeed,
    TVector<TSumMulti>* leafDersSum
);

void AddLangevinNoiseToLeafNewtonSum(
    float diffusionTemperature,
    float learningRate,
    double scaledL2Regularizer,
    ui64 randomSeed,
    TVector<TSum>* leafDersSum
);

void AddLangevinNoiseToLeafNewtonSum(
    float diffusionTemperature,
    float learningRate,
    double scaledL2Regularizer,
    ui64 randomSeed,
    TVector<TSumMulti>* leafDersSum
);
