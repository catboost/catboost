#include "sample.h"

#include <util/system/yassert.h>
#include <util/generic/vector.h>
#include <util/generic/array_ref.h>

using NMetrics::TSample;

TVector<TSample> TSample::FromVectors(
    const TConstArrayRef<double> targets,
    const TConstArrayRef<double> predictions)
{
    Y_ASSERT(targets.size() == predictions.size());
    TVector<TSample> result;
    result.reserve(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        result.emplace_back(targets[i], predictions[i]);
    }
    return result;
}

TVector<TSample> TSample::FromVectors(
    const TConstArrayRef<double> targets,
    const TConstArrayRef<double> predictions,
    const TConstArrayRef<double> weights)
{
    Y_ASSERT(targets.size() == predictions.size());
    Y_ASSERT(targets.size() == weights.size());
    TVector<TSample> result;
    result.reserve(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        result.emplace_back(targets[i], predictions[i], weights[i]);
    }
    return result;
}
