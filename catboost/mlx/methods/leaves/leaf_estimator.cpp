#include "leaf_estimator.h"

namespace NCatboostMlx {

    mx::array ComputeLeafValues(
        const mx::array& gradientSums,
        const mx::array& hessianSums,
        float l2RegLambda,
        float learningRate
    ) {
        // Newton step: -gradSum / (hessSum + lambda) * lr
        // Using MLX ops (no custom kernel needed — numLeaves is small, typically 2-256)
        auto denominator = mx::add(hessianSums, mx::array(l2RegLambda));
        auto rawValues = mx::negative(mx::divide(gradientSums, denominator));
        auto leafValues = mx::multiply(rawValues, mx::array(learningRate));

        TMLXDevice::EvalNow(leafValues);
        return leafValues;
    }

}  // namespace NCatboostMlx
