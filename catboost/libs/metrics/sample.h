#pragma once

#include <util/generic/vector.h>

namespace NMetrics {
struct TSample {
    double Target;
    double Prediction;
    double Weight;

    TSample(double target, double prediction, double weight = 1)
        : Target(target)
        , Prediction(prediction)
        , Weight(weight)
    {}

    static TVector<TSample> FromVectors(
        const TVector<double>& targets, const TVector<double>& predictions);
    static TVector<TSample> FromVectors(
        const TVector<double>& targets, const TVector<double>& predictions, const TVector<double>& weights);
};
}  // NMetrics
