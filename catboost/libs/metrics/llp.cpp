#include "metric_holder.h"

#include <catboost/libs/eval_result/eval_helpers.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>

#include <util/generic/string.h>

TMetricHolder CalcLlp(TConstArrayRef<double> approx,
                      TConstArrayRef<float> target,
                      TConstArrayRef<float> weight,
                      int begin,
                      int end) {
    TMetricHolder metric(3); // metric.Stats[0] = result, metric.Stats[1] = clicks, metric.Stats[2] = shows
    for (int i = begin; i < end; ++i) {
        const float w = weight.empty() ? 1 : weight[i];
        const float expVal = exp(approx[i]);
        const float logExpPlusOne = IsFinite(expVal) ? log(1 + expVal) : static_cast<float>(approx[i]);
        metric.Stats[0] += (target[i] * approx[i] - logExpPlusOne) * w;
        metric.Stats[1] += target[i] * w;
        metric.Stats[2] += w;
    }
    return metric;
}

double CalcLlp(const TMetricHolder& error) {
    double ll = error.Stats[0];
    double clicks = error.Stats[1];
    double shows = error.Stats[2];
    if (clicks > 0 && shows > clicks) {
        double ctr0 = shows != 0 ? clicks / shows : 0;
        double ll0 = clicks * log(ctr0) + (shows - clicks) * log(1 - ctr0);
        return clicks != 0 ? (ll - ll0) / clicks : 0;
    }
    return 0;
}
