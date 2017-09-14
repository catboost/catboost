#pragma once

#include <library/fast_exp/fast_exp.h>
#include <library/fast_log/fast_log.h>

#include <util/generic/vector.h>
#include <util/generic/ymath.h>

template<bool StoreExpApprox>
static inline double UpdateApprox(double approx, double approxDelta) {
    return StoreExpApprox ? approx * approxDelta : approx + approxDelta;
}

template<bool StoreExpApprox>
static inline void ExpApproxIf(yvector<double>* approx) {
    if (StoreExpApprox) {
        FastExpInplace(approx->data(), approx->ysize());
    }
}

template<bool StoreExpApprox>
static inline double GetNeutralApprox() {
    return StoreExpApprox ? 1.0 : 0.0;
}

template<bool StoreExpApprox>
static inline double ApplyLearningRate(double approxDelta, double learningRate) {
    return StoreExpApprox ? fast_exp(FastLogf(approxDelta) * learningRate) : approxDelta * learningRate;
}

inline double GetNeutralApprox(bool storeExpApproxes) {
    if (storeExpApproxes) {
        return GetNeutralApprox</*StoreExpApprox*/ true>();
    } else {
        return GetNeutralApprox</*StoreExpApprox*/ false>();
    }
}

inline void ExpApproxIf(bool storeExpApproxes, yvector<yvector<double>>* approxMulti) {
    if (storeExpApproxes) {
        for (auto& approx : *approxMulti) {
            ExpApproxIf</*StoreExpApprox*/ true>(&approx);
        }
    } else {
        for (auto& approx: *approxMulti) {
            ExpApproxIf</*StoreExpApprox*/ false>(&approx);
        }
    }
}
