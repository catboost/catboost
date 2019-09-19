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

        static void FromVectors(
            TConstArrayRef<float> targets,
            TConstArrayRef<double> predictions,
            TVector<TSample>* samples);

        static TVector<TSample> FromVectors(
            TConstArrayRef<float> targets,
            TConstArrayRef<double> predictions);

        static void FromVectors(
            TConstArrayRef<double> targets,
            TConstArrayRef<double> predictions,
            TVector<TSample>* samples);

        static TVector<TSample> FromVectors(
            TConstArrayRef<double> targets,
            TConstArrayRef<double> predictions);

        static void FromVectors(
            TConstArrayRef<double> targets,
            TConstArrayRef<double> predictions,
            TConstArrayRef<double> weights,
            TVector<TSample>* samples);

        static TVector<TSample> FromVectors(
            TConstArrayRef<double> targets,
            TConstArrayRef<double> predictions,
            TConstArrayRef<double> weights);
    };

    struct TBinClassSample {
        double Prediction = 0;
        double Weight = 0;

        TBinClassSample() = default;

        TBinClassSample(double prediction, double weight = 1)
            : Prediction(prediction)
            , Weight(weight)
        {}
    };
}  // NMetrics
