#include "metric_holder.h"

#include <catboost/libs/model/eval_processing.h>

#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/generic/array_ref.h>

static TMetricHolder ComputeBrierScoreMetric(TConstArrayRef<double> approxes,
                                      TConstArrayRef<float> targets,
                                      TConstArrayRef<float> weights) {
    Y_ASSERT(approxes.size() == targets.size());
    Y_ASSERT(weights.empty() || (approxes.size() == weights.size()));

    TMetricHolder error(2);

    double score = 0;
    double sum = 0;
    for (size_t i = 0; i < approxes.size(); ++i) {
        float weight = weights.empty() ? 1.0f : weights[i];
        score += Sqr(targets[i] - approxes[i]) * weight;
        sum += weight;
    }

    error.Stats[0] = score;
    error.Stats[1] = sum;

    return error;
}

TMetricHolder ComputeBrierScoreMetric(TConstArrayRef<double> approxes,
                               TConstArrayRef<float> targets,
                               TConstArrayRef<float> weights,
                               const int begin,
                               const int end) {
    TConstArrayRef<double> partOfApproxes = approxes.Slice(begin, end - begin);
    return ComputeBrierScoreMetric(
            CalcSigmoid(TVector<double>(partOfApproxes.begin(), partOfApproxes.end())),
            targets.Slice(begin, end - begin),
            weights.empty() ? TConstArrayRef<float>() : weights.Slice(begin, end - begin));
}
