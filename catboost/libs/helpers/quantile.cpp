#include "quantile.h"

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>

#include <algorithm>

namespace {
    struct TValueWithWeight {
        float Value;
        float Weight;
    };
}

static double CalcSampleQuantileBinarySearch(
    TConstArrayRef<float> sampleRef,
    TConstArrayRef<float> weightsRef,
    const double alpha
) {
    constexpr int BINARY_SEARCH_ITERATIONS = 100;

    const double totalWeight = Accumulate(weightsRef, 0.0);
    const double needWeight = totalWeight * alpha;

    const auto minMaxSamples = MinMaxElement(sampleRef.begin(), sampleRef.end());
    double lQ = *minMaxSamples.first - DBL_EPSILON;
    double rQ = *minMaxSamples.second;

    const size_t sampleSize = sampleRef.size();
    TVector<TValueWithWeight> elements;
    elements.yresize(sampleSize);
    for (auto i : xrange(sampleSize)) {
        elements[i] = {sampleRef[i], weightsRef[i]};
    }
    /*
     * We will support the following invariant:
     * total weight of elements with sample values <= lQ is strictly less than needWeight
     * total weight of elements with sample values <= rQ is greater or equal than needWeight
     * elements with indices < l are less or equal than q
     * elements with indices > r are greater than q
     */
    int l = 0, r = sampleSize;
    double collectedLeftWeight = 0;
    for (auto it : xrange(BINARY_SEARCH_ITERATIONS)) {
        Y_UNUSED(it);
        const double q = (lQ + rQ) / 2;
        auto partitionIt = std::partition(
            elements.begin() + l,
            elements.begin() + r,
            [q](const TValueWithWeight& element) {
                return element.Value <= q;
            }
        );
        const double partitionLeftWeight = Accumulate(
            elements.begin() + l,
            partitionIt,
            0.0,
            [](double sum, const TValueWithWeight& element) {
                return sum + element.Weight;
            }
        );
        const int partitionPoint = partitionIt - elements.begin();

        if (collectedLeftWeight + partitionLeftWeight < needWeight - DBL_EPSILON) {
            l = partitionPoint;
            lQ = q;
            collectedLeftWeight += partitionLeftWeight;
        } else {
            r = partitionPoint;
            rQ = q;
        }
    }
    return rQ;
}

static double CalcSampleQuantileLinearSearch(
    TConstArrayRef<float> sampleRef,
    TConstArrayRef<float> weightsRef,
    const double alpha
) {
    const int sampleSize = sampleRef.size();
    TVector<TValueWithWeight> elements;
    elements.yresize(sampleSize);
    for (auto i : xrange(sampleSize)) {
        elements[i] = {sampleRef[i], weightsRef[i]};
    }
    StableSort(elements, [](const TValueWithWeight& elem1, const TValueWithWeight& elem2) {
        return elem1.Value < elem2.Value;
    });
    const double totalWeight = Accumulate(weightsRef, 0.0);
    const double needWeight = totalWeight * alpha;
    double sumWeight = 0;
    for (const auto& element : elements) {
        sumWeight += element.Weight;
        if (sumWeight >= needWeight - DBL_EPSILON) {
            return element.Value;
        }
    }
    return elements.back().Value;
}

double CalcSampleQuantile(
    TConstArrayRef<float> sampleRef,
    TConstArrayRef<float> weightsRef,
    const double alpha
) {
    if (sampleRef.empty()) {
        return 0.0;
    }
    if (alpha <= 0) {
        return *MinElement(sampleRef.begin(), sampleRef.end());
    }
    Y_ASSERT(0 <= alpha && alpha <= 1);
    TVector<float> defaultWeights;
    if (weightsRef.empty()) {
        defaultWeights.resize(sampleRef.size(), 1.0);
        weightsRef = defaultWeights;
    }
    Y_ASSERT(sampleRef.size() == weightsRef.size());
    return sampleRef.size() < 100
        ? CalcSampleQuantileLinearSearch(sampleRef, weightsRef, alpha)
        : CalcSampleQuantileBinarySearch(sampleRef, weightsRef, alpha);
}
