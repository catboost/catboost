#pragma once

#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/data_types/query.h>

#include <library/fast_exp/fast_exp.h>
#include <library/fast_log/fast_log.h>

#include <util/generic/vector.h>
#include <util/generic/ymath.h>

template<bool StoreExpApprox>
static inline double UpdateApprox(double approx, double approxDelta) {
    return StoreExpApprox ? approx * approxDelta : approx + approxDelta;
}

template<bool StoreExpApprox>
static inline double GetNeutralApprox() {
    return StoreExpApprox ? 1.0 : 0.0;
}

template<bool StoreExpApprox>
static inline double ApplyLearningRate(double approxDelta, double learningRate) {
    return StoreExpApprox ? fast_exp(FastLogf(approxDelta) * learningRate) : approxDelta * learningRate;
}

static inline double GetNeutralApprox(bool storeExpApproxes) {
    if (storeExpApproxes) {
        return GetNeutralApprox</*StoreExpApprox*/ true>();
    } else {
        return GetNeutralApprox</*StoreExpApprox*/ false>();
    }
}

static inline void ExpApproxIf(bool storeExpApproxes, TVector<double>* approx) {
    if (storeExpApproxes) {
        FastExpInplace(approx->data(), approx->ysize());
    }
}

static inline void ExpApproxIf(bool storeExpApproxes, TVector<TVector<double>>* approxMulti) {
    for (auto& approx : *approxMulti) {
        ExpApproxIf(storeExpApproxes, &approx);
    }
}


inline bool IsStoreExpApprox(ELossFunction lossFunction) {
    return EqualToOneOf(
        lossFunction,
        ELossFunction::Logloss,
        ELossFunction::LogLinQuantile,
        ELossFunction::Poisson,
        ELossFunction::CrossEntropy,
        ELossFunction::PairLogit,
        ELossFunction::YetiRank
    );
}

inline void CalcPairwiseWeights(const TVector<TQueryInfo>& queriesInfo, int queriesCount, TVector<float>* pairwiseWeights) {
    Fill(pairwiseWeights->begin(), pairwiseWeights->end(), 0);
    for (int queryIndex = 0; queryIndex < queriesCount; ++queryIndex) {
        const auto& queryInfo = queriesInfo[queryIndex];
        for (int docId = 0; docId < queryInfo.Competitors.ysize(); ++docId) {
            for (const auto& competitor : queryInfo.Competitors[docId]) {
                (*pairwiseWeights)[queryInfo.Begin + docId] += competitor.Weight;
                (*pairwiseWeights)[queryInfo.Begin + competitor.Id] += competitor.Weight;
            }
        }
    }
}
