#pragma once

#include <util/generic/fwd.h>

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
            TConstArrayRef<double> targets,
            TConstArrayRef<double> predictions);

        static TVector<TSample> FromVectors(
            TConstArrayRef<double> targets,
            TConstArrayRef<double> predictions,
            TConstArrayRef<double> weights);
    };
}  // NMetrics
