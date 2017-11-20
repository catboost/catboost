#include "sample.h"

#include <util/system/yassert.h>

using NMetrics::TSample;

TVector<TSample> TSample::FromVectors(const TVector<double>& targets, const TVector<double>& predictions) {
    Y_ASSERT(targets.size() == predictions.size());
    TVector<TSample> result;
    result.reserve(targets.size());
    for (ui32 i = 0; i < targets.size(); ++i) {
        result.push_back(TSample(targets[i], predictions[i]));
    }
    return result;
}

TVector<TSample> TSample::FromVectors(
    const TVector<double>& targets, const TVector<double>& predictions, const TVector<double>& weights)
{
    Y_ASSERT(targets.size() == predictions.size());
    Y_ASSERT(targets.size() == weights.size());
    TVector<TSample> result;
    result.reserve(targets.size());
    for (ui32 i = 0; i < targets.size(); ++i) {
        result.push_back(TSample(targets[i], predictions[i], weights[i]));
    }
    return result;
}
