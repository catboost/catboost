#include "hinge_loss.h"

#include "metric_holder.h"

#include <util/generic/vector.h>
#include <util/generic/array_ref.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>

#include <limits>


TMetricHolder ComputeHingeLossMetric(TConstArrayRef<TConstArrayRef<double>> approx,
                                     TConstArrayRef<float> target,
                                     TConstArrayRef<float> weight,
                                     int begin,
                                     int end,
                                     double targetBorder) {
    TMetricHolder error(2);
    error.Stats[0] = 0;
    error.Stats[1] = 0;

    const bool isMulticlass = approx.size() > 1;

    for (auto index : xrange(begin, end)) {
        double value;
        float w = weight.empty() ? 1 : weight[index];

        if (isMulticlass) {
            auto targetValue = static_cast<size_t>(target[index]);
            double maxApprox = std::numeric_limits<double>::lowest();

            for (auto j : xrange(approx.size())) {
                if (targetValue != j) {
                    maxApprox = Max(maxApprox, approx[j][index]);
                }
            }
            value = 1 - (approx[targetValue][index] - maxApprox);
        } else {
            if (target[index] > targetBorder) {;
                value = 1 - approx.front()[index];
            } else {
                value = 1 + approx.front()[index];
            }
        }

        error.Stats[0] += (value < 0 ? 0 : value) * w;
        error.Stats[1] += w;
    }
    return error;
}
