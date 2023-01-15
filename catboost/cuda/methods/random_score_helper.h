#pragma once

#include <cmath>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/cuda_util/dot_product.h>

namespace NCatboostCuda {
    template <class TTarget>
    inline double ComputeStdDev(const TTarget& target) {
        auto tmp = decltype(target.WeightedTarget)::CopyMapping(target.WeightedTarget);
        tmp.Copy(target.WeightedTarget);
        DivideVector(tmp, target.Weights);
        const double sum2 = DotProduct(tmp, tmp, &target.Weights);
        const double count = target.WeightedTarget.GetObjectsSlice().Size();
        return sqrt(sum2 / (count + 1e-100));
    }

    inline double CalcScoreModelLengthMult(const double sampleCount, double modelSize) {
        double modelExpLength = log(sampleCount);
        double modelLeft = exp(modelExpLength - modelSize);
        return modelLeft / (1 + modelLeft);
    }

    template <class TTarget>
    inline double ComputeScoreStdDev(double modelLengthMult, double randomStrength, const TTarget& target) {
        if (modelLengthMult * randomStrength) {
            double stdDev = ComputeStdDev(target);
            return modelLengthMult * stdDev * randomStrength;
        } else {
            return 0;
        }
    }

}
