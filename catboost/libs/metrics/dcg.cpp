#include "dcg.h"
#include "doc_comparator.h"
#include "sample.h"

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>

using NMetrics::TSample;

static double CalcDcgSorted(TConstArrayRef<TSample> samples, ENdcgMetricType type, TMaybe<double> expDecay = Nothing()) {
    double result = 0;
    double decay = 1;
    for (size_t i = 0; i < samples.size(); ++i) {
        Y_ASSERT(samples[i].Weight == 1);
        if (expDecay.Defined()) {
            if (i > 0) {
                decay *= *expDecay;
            }
        } else {
            decay = 1 / Log2(2 + i);
        }

        const double value = (type == ENdcgMetricType::Base) ? samples[i].Target : (pow(2, samples[i].Target) - 1);
        result += decay * value;
    }
    return result;
}

double CalcDcg(TConstArrayRef<TSample> samplesRef, ENdcgMetricType type, TMaybe<double> expDecay) {
    TVector<TSample> samples(samplesRef.begin(), samplesRef.end());
    Sort(samples.begin(), samples.end(), [](const TSample& left, const TSample& right) {
        return CompareDocs(left.Prediction, left.Target, right.Prediction, right.Target);
    });
    return CalcDcgSorted(samples, type, expDecay);
}

double CalcIDcg(TConstArrayRef<TSample> samplesRef, ENdcgMetricType type, TMaybe<double> expDecay) {
    TVector<TSample> samples(samplesRef.begin(), samplesRef.end());

    Sort(samples.begin(), samples.end(), [](const TSample& left, const TSample& right) {
        return left.Target > right.Target;
    });
    return CalcDcgSorted(samples, type, expDecay);
}


double CalcNdcg(TConstArrayRef<TSample> samples, ENdcgMetricType type) {
    double dcg = CalcDcg(samples, type);
    double idcg = CalcIDcg(samples, type);
    return idcg > 0 ? dcg / idcg : 0;
}
