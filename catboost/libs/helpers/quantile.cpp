#include "quantile.h"

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>


double CalcSampleQuantileWithIndices(
    TConstArrayRef<float> sample,
    TConstArrayRef<float> weights,
    TConstArrayRef<size_t> indices,
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
        return *MinElement(sample.begin(), sample.end()) - delta;
    }
    if (doublePosition >= sumWeight) {
        return *MaxElement(sample.begin(), sample.end()) + delta;
    }

    size_t position = 0;
    float sum = 0;
    while (sum < doublePosition && position < sampleSize) {
        size_t j = position;
        float step = 0;
        while (j < sampleSize && Abs(sample[indices[j]] - sample[indices[position]]) < DBL_EPSILON) {
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

double CalcSampleQuantile(
    TConstArrayRef<float> sample,
    TConstArrayRef<float> weights,
    const double alpha,
    const double delta
) {
    size_t sampleSize = sample.size();

    TVector<size_t> indices(sampleSize);
    Iota(indices.begin(), indices.end(), 0);

    Sort(indices.begin(), indices.end(), [&](size_t i, size_t j) { return sample[i] < sample[j]; });

    return CalcSampleQuantileWithIndices(sample, weights, indices, alpha, delta);
}

double CalcSampleQuantileSorted(
    TConstArrayRef<float> sample,
    TConstArrayRef<float> weights,
    const double alpha,
    const double delta
) {
    size_t sampleSize = sample.size();

    TVector<size_t> indices(sampleSize);
    Iota(indices.begin(), indices.end(), 0);

    return CalcSampleQuantileWithIndices(sample, weights, indices, alpha, delta);
}
