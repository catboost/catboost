#pragma once

#include <util/generic/fwd.h>

namespace NMetrics {
    struct TSample {
        double Target = 0;
        double Prediction = 0;
        double Weight = 0;

        TSample() = default;

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
