#include "langevin_utils.h"

#include <catboost/private/libs/algo_helpers/ders_holder.h>
#include <catboost/private/libs/algo_helpers/online_predictor.h>
#include <catboost/private/libs/index_range/index_range.h>
#include <catboost/private/libs/options/restrictions.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/random/fast.h>
#include <util/random/normal.h>

using namespace NCB;

double CalcLangevinNoiseRate(float diffusionTemperature, float learningRate) {
    return sqrt(2.0 / learningRate / diffusionTemperature);
}

void AddLangevinNoiseToDerivatives(
    float diffusionTemperature,
    float learningRate,
    ui64 randomSeed,
    TVector<double>* derivatives,
    NPar::ILocalExecutor* localExecutor
) {
    if (diffusionTemperature == 0.0f) {
        return;
    }
    const double coef = CalcLangevinNoiseRate(diffusionTemperature, learningRate);
    CB_ENSURE_INTERNAL(!derivatives->empty(), "Unexpected empty derivatives");
    const size_t objectCount = derivatives->size();
    TSimpleIndexRangesGenerator<size_t> rangesGenerator(TIndexRange(objectCount), CB_THREAD_LIMIT);

    localExecutor->ExecRange(
        [&](int blockIdx) {
            TFastRng64 blockRng(randomSeed + blockIdx);
            auto dersData = derivatives->data();
            for (auto idx : rangesGenerator.GetRange(blockIdx).Iter()) {
                dersData[idx] += coef * StdNormalDistribution<double>(blockRng);
            }
        },
        0,
        SafeIntegerCast<int>(rangesGenerator.RangesCount()),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}

void AddLangevinNoiseToDerivatives(
    float diffusionTemperature,
    float learningRate,
    ui64 randomSeed,
    TVector<TVector<double>>* derivatives,
    NPar::ILocalExecutor* localExecutor
) {
    if (diffusionTemperature == 0.0f) {
        return;
    }
    const double coef = CalcLangevinNoiseRate(diffusionTemperature, learningRate);
    CB_ENSURE_INTERNAL(!derivatives->empty(), "Unexpected empty derivatives");
    const size_t objectCount = derivatives->front().size();
    TSimpleIndexRangesGenerator<size_t> rangesGenerator(TIndexRange(objectCount), CB_THREAD_LIMIT);
    for(auto& derivatives1d : *derivatives) {
        localExecutor->ExecRange(
            [&](int blockIdx) {
                TFastRng64 blockRng(randomSeed + blockIdx);
                auto dersData = derivatives1d.data();
                for (auto idx : rangesGenerator.GetRange(blockIdx).Iter()) {
                    dersData[idx] += coef * StdNormalDistribution<double>(blockRng);
                }
            },
            0,
            SafeIntegerCast<int>(rangesGenerator.RangesCount()),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }
}

void AddLangevinNoiseToLeafDerivativesSum(
    float diffusionTemperature,
    float learningRate,
    double scaledL2Regularizer,
    ui64 randomSeed,
    TVector<TSum>* leafDersSum
) {
    if (diffusionTemperature == 0.0f) {
        return;
    }
    TFastRng64 rng(randomSeed);
    const double coef = CalcLangevinNoiseRate(diffusionTemperature, learningRate);
    for (TSum& sum : *leafDersSum) {
        if (sum.SumWeights < 1e-9) {
            continue;
        }
        double scaledCoef = coef * sqrt(sum.SumWeights + scaledL2Regularizer);
        sum.SumDer += scaledCoef * StdNormalDistribution<double>(rng);
    }
}

void AddLangevinNoiseToLeafDerivativesSum(
    float diffusionTemperature,
    float learningRate,
    double scaledL2Regularizer,
    ui64 randomSeed,
    TVector<TSumMulti>* leafDersSum
) {
    if (diffusionTemperature == 0.0f) {
        return;
    }
    TFastRng64 rng(randomSeed);
    const double coef = CalcLangevinNoiseRate(diffusionTemperature, learningRate);
    for (TSumMulti& sum : *leafDersSum) {
        if (sum.SumWeights < 1e-9) {
            continue;
        }
        double scaledCoef = coef * sqrt(sum.SumWeights + scaledL2Regularizer);
        for (auto& der : sum.SumDer) {
            der += scaledCoef * StdNormalDistribution<double>(rng);
        }
    }
}

void AddLangevinNoiseToLeafNewtonSum(
    float diffusionTemperature,
    float learningRate,
    double scaledL2Regularizer,
    ui64 randomSeed,
    TVector<TSum>* leafDersSum
) {
    if (diffusionTemperature == 0.0f) {
        return;
    }
    TFastRng64 rng(randomSeed);
    const double coef = CalcLangevinNoiseRate(diffusionTemperature, learningRate);
    for (TSum& sum : *leafDersSum) {
        if (sum.SumWeights < 1e-9) {
            continue;
        }
        double scaledCoef = coef * sqrt(std::fabs(sum.SumDer2) + scaledL2Regularizer);
        sum.SumDer += scaledCoef * StdNormalDistribution<double>(rng);
    }
}

void AddLangevinNoiseToLeafNewtonSum(
    float diffusionTemperature,
    float learningRate,
    double scaledL2Regularizer,
    ui64 randomSeed,
    TVector<TSumMulti>* leafDersSum
) {
    if (diffusionTemperature == 0.0f) {
        return;
    }
    TFastRng64 rng(randomSeed);
    const double coef = CalcLangevinNoiseRate(diffusionTemperature, learningRate);
    for (TSumMulti& sum : *leafDersSum) {
        if (sum.SumWeights < 1e-9) {
            continue;
        }
        for (int i = 0; i < sum.SumDer.ysize(); ++i) {
            double scaledCoef = coef * sqrt(std::fabs(sum.SumDer2.Data[i]) + scaledL2Regularizer);
            sum.SumDer[i] += scaledCoef * StdNormalDistribution<double>(rng);
        }
    }
}

