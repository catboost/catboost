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

    static yvector<TSample> FromVectors(
        const yvector<double>& targets, const yvector<double>& predictions);
    static yvector<TSample> FromVectors(
        const yvector<double>& targets, const yvector<double>& predictions, const yvector<double>& weights);
};
}  // NMetrics
