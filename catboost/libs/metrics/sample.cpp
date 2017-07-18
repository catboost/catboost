#include "sample.h"

#include <util/system/yassert.h>

using NMetrics::TSample;

yvector<TSample> TSample::FromVectors(const yvector<double>& targets, const yvector<double>& predictions) {
    Y_ASSERT(targets.size() == predictions.size());
    yvector<TSample> result;
    result.reserve(targets.size());
    for (ui32 i = 0; i < targets.size(); ++i) {
        result.push_back(TSample(targets[i], predictions[i]));
    }
    return result;
}

yvector<TSample> TSample::FromVectors(
    const yvector<double>& targets, const yvector<double>& predictions, const yvector<double>& weights)
{
    Y_ASSERT(targets.size() == predictions.size());
    Y_ASSERT(targets.size() == weights.size());
    yvector<TSample> result;
    result.reserve(targets.size());
    for (ui32 i = 0; i < targets.size(); ++i) {
        result.push_back(TSample(targets[i], predictions[i], weights[i]));
    }
    return result;
}
