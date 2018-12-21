#include "sample.h"

#include <util/system/yassert.h>
#include <util/generic/vector.h>
#include <util/generic/array_ref.h>

using NMetrics::TSample;

void TSample::FromVectors(
    const TConstArrayRef<float> targets,
    const TConstArrayRef<double> predictions,
    TVector<TSample>* const samples)
{
    const auto defaultWeight = TSample(0.f, 0.f).Weight;
    Y_ASSERT(targets.size() == predictions.size());
    samples->yresize(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        auto& sample = (*samples)[i];
        sample.Target = targets[i];
        sample.Prediction = predictions[i];
        sample.Weight = defaultWeight;
    }
}

TVector<TSample> TSample::FromVectors(
    const TConstArrayRef<float> targets,
    const TConstArrayRef<double> predictions)
{
    TVector<TSample> samples;
    FromVectors(targets, predictions, &samples);
    return samples;
}

void TSample::FromVectors(
    const TConstArrayRef<double> targets,
    const TConstArrayRef<double> predictions,
    TVector<TSample>* const samples)
{
    const auto defaultWeight = TSample(0.f, 0.f).Weight;
    Y_ASSERT(targets.size() == predictions.size());
    samples->yresize(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        auto& sample = (*samples)[i];
        sample.Target = targets[i];
        sample.Prediction = predictions[i];
        sample.Weight = defaultWeight;
    }
}

TVector<TSample> TSample::FromVectors(
    const TConstArrayRef<double> targets,
    const TConstArrayRef<double> predictions)
{
    TVector<TSample> samples;
    FromVectors(targets, predictions, &samples);
    return samples;
}

void TSample::FromVectors(
    const TConstArrayRef<double> targets,
    const TConstArrayRef<double> predictions,
    const TConstArrayRef<double> weights,
    TVector<TSample>* const samples)
{
    Y_ASSERT(targets.size() == predictions.size());
    Y_ASSERT(targets.size() == weights.size());
    samples->yresize(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        auto& sample = (*samples)[i];
        sample.Target = targets[i];
        sample.Prediction = predictions[i];
        sample.Weight = weights[i];
    }
}

TVector<TSample> TSample::FromVectors(
    const TConstArrayRef<double> targets,
    const TConstArrayRef<double> predictions,
    const TConstArrayRef<double> weights)
{
    TVector<TSample> samples;
    FromVectors(targets, predictions, weights, &samples);
    return samples;
}
