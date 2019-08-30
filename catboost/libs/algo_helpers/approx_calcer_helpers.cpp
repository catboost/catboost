#include "approx_calcer_helpers.h"

double CalcSampleQuantile(
    TConstArrayRef<double> sample,
    TConstArrayRef<double> weights,
    const double alpha,
    const double delta
) {
    if (sample.empty()) {
        return 0;
    }
    size_t sampleSize = sample.size();

    double sumWeight = 0;
    for (size_t i = 0; i < sampleSize; i++) {
        sumWeight += weights[i];
    }
    double doublePosition = sumWeight * alpha;
    if (doublePosition <= 0) {
        return *std::min_element(sample.begin(), sample.end()) - delta;
    }
    if (doublePosition >= sumWeight) {
        return *std::max_element(sample.begin(), sample.end()) + delta;
    }

    TVector<size_t> indices(sampleSize);
    for (size_t i = 0; i < sampleSize; i++) {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) { return sample[i] < sample[j]; });

    size_t position = 0;
    float sum = 0;
    while (sum < doublePosition && position < sampleSize) {
        size_t j = position;
        float step = 0;
        while (j < sampleSize && abs(sample[indices[j]] - sample[indices[position]]) < DBL_EPSILON) {
            step += weights[indices[j]];
            j++;
        }
        if (sum + alpha * step >= doublePosition - DBL_EPSILON) {
            return sample[indices[position]] - delta;
        }
        if (sum + step >= doublePosition + DBL_EPSILON) {
            return sample[indices[position]] + delta;
        }

        sum += step;
        position = j;

        if (sum >= doublePosition - DBL_EPSILON) {
            if (position >= sampleSize) {
                return sample[indices[position - 1]] + delta;
            }
            return (sample[indices[position - 1]] + delta) * alpha + (sample[indices[position]] - delta) * (1 - alpha);
        }

    }
    Y_ASSERT(false);
    return 0;
}

void CreateBacktrackingObjective(
    int dimensionCount,
    int leavesEstimationIterations,
    ELeavesEstimationStepBacktracking leavesEstimationBacktrackingType,
    const NCatboostOptions::TLossDescription& objectiveMetric,
    bool* haveBacktrackingObjective,
    double* minimizationSign,
    TVector<THolder<IMetric>>* lossFunction
) {
    *haveBacktrackingObjective = leavesEstimationIterations > 1
                                 && leavesEstimationBacktrackingType != ELeavesEstimationStepBacktracking::No;
    if (*haveBacktrackingObjective) {
        *lossFunction = CreateMetricFromDescription(objectiveMetric, dimensionCount);
        *minimizationSign = GetMinimizeSign((*lossFunction)[0]);
    }
}

double GetMinimizeSign(const THolder<IMetric>& metric) {
    EMetricBestValue bestMetric;
    float ignoredBestValue;
    metric->GetBestValue(&bestMetric, &ignoredBestValue);
    switch (bestMetric) {
        case EMetricBestValue::Min: {
            return 1.0;
        }
        case EMetricBestValue::Max: {
            return -1.0;
        }
        default: {
            Y_UNREACHABLE();
        }
    }
}
