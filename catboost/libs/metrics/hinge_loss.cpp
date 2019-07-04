#include "metric_holder.h"

#include <catboost/libs/model/eval_processing.h>

#include <util/generic/vector.h>
#include <util/generic/array_ref.h>

#include <cmath>

TMetricHolder ComputeHingeLossMetric(TConstArrayRef<TVector<double>> approx,
                                     TConstArrayRef<float> target,
                                     TConstArrayRef<float> weight,
                                     int begin,
                                     int end) {
    TMetricHolder error(2);
    error.Stats[0] = 0;
    error.Stats[1] = 0;

    const bool isMulticlass = approx.size() > 1;

    for (int index = begin; index < end; ++index) {
        double value;
        float w = weight.empty() ? 1 : weight[index];

        if (isMulticlass) {
            auto targetValue = static_cast<size_t>(target[index]);
            double maxApprox = (targetValue == 0 ? approx[1][index] : approx[0][index]);

            for (size_t j = 0; j < approx.size(); ++j) {
                if (targetValue != j && approx[j][index] > maxApprox) {
                    maxApprox = approx[j][index];
                }
            }
            value = 1 - (approx[targetValue][index] - maxApprox);
        } else {
            TVector<double> probabilities = CalcSigmoid(approx.front());
            if (target[index]) {;
                value = 1 - probabilities[index];
            } else {
                value = 1 + probabilities[index];
            }
        }

        error.Stats[0] += (value < 0 ? 0 : value) * w;
        error.Stats[1] += w;
    }
    return error;
}
