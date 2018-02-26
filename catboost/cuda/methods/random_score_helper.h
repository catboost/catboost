#pragma once

#include <cmath>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/cuda_util/dot_product.h>

namespace NCatboostCuda {

    template <class TTarget>
    inline double ComputeStdDev(TTarget& target) {
        DivideVector(target.WeightedTarget, target.Weights);
        const double sum2 = DotProduct(target.WeightedTarget, target.WeightedTarget, &target.Weights);
        const double count = target.WeightedTarget.GetObjectsSlice().Size();
        MultiplyVector(target.WeightedTarget, target.Weights);
        return sqrt(sum2 / (count + 1e-100));
    }

    inline double CalcScoreModelLengthMult(const double sampleCount, double modelSize) {
        double modelExpLength = log(sampleCount);
        double modelLeft = exp(modelExpLength - modelSize);
        return modelLeft / (1 + modelLeft);
    }

    template <class TTarget>
    inline double ComputeScoreStdDev(double modelLengthMult, double randomStrength, TTarget& target) {

        if (modelLengthMult * randomStrength) {
            double stdDev = ComputeStdDev(target);
            return modelLengthMult * stdDev * randomStrength;
        } else {
            return 0;
        }
    }

}
