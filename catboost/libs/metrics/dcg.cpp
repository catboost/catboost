#include "dcg.h"

#include <util/generic/ymath.h>
#include <util/generic/algorithm.h>
#include "doc_comparator.h"

using NMetrics::TSample;

namespace {
    double CalcDCGSorted(const TVector<TSample>& samples, TMaybe<double> expDecay) {
        double result = 0;
        double decay = 1;
        for (ui32 i = 0; i < samples.size(); ++i) {
            Y_ASSERT(samples[i].Weight == 1);
            if (expDecay.Defined()) {
                if (i > 0) {
                    decay *= *expDecay;
                }
            } else {
                decay = 1 / log2(2 + i);
            }
            double value = samples[i].Target;
            result += decay * value;
        }
        return result;
    }
}

double CalcDCG(TVector<TSample>* samples, TMaybe<double> expDecay) {
    Sort(samples->begin(), samples->end(), [](const TSample& left, const TSample& right) {
        return CompareDocs(left.Prediction, right.Target, right.Prediction, left.Target);
    });
    auto optimisticDCG = CalcDCGSorted(*samples, expDecay);
    Sort(samples->begin(), samples->end(), [](const TSample& left, const TSample& right) {
        return CompareDocs(left.Prediction, left.Target, right.Prediction, right.Target);
    });
    auto pessimisticDCG = CalcDCGSorted(*samples, expDecay);
    return (optimisticDCG + pessimisticDCG) / 2;
}

double CalcIDCG(TVector<TSample>* samples, TMaybe<double> expDecay) {
    Sort(samples->begin(), samples->end(), [](const TSample& left, const TSample& right) {
        return left.Target > right.Target;
    });
    return CalcDCGSorted(*samples, expDecay);
}


double CalcNDCG(TVector<TSample>* samples) {
    double dcg = CalcDCG(samples);
    double idcg = CalcIDCG(samples);

    return idcg > 0 ? dcg / idcg : 0;
}
