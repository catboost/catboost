#include "quantile.h"

#include <catboost/private/libs/algo_helpers/quantile_selection.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>

#include <algorithm>
#include <cmath>

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

static double CalcTargetDependentMinimumBinarySearch(
    TConstArrayRef<float> sampleRef,
    TConstArrayRef<float> weightsRef,
    TConstArrayRef<float> origTarget,
    TConstArrayRef<double> boundaries,
    TConstArrayRef<double> quantiles)
{
    constexpr int BINARY_SEARCH_ITERATIONS = 100;
    const size_t sampleSize = sampleRef.size();

    TVector<TValueWithWeight> elements;
    TVector<TValueWithWeight> origs;
    elements.yresize(sampleSize);
    origs.yresize(sampleSize);
    for (auto i : xrange(sampleSize))
    {
        elements[i] = {sampleRef[i], weightsRef[i]};
        origs[i] = {sampleRef[i], origTarget[i]};
    }
    Sort(elements, [](const TValueWithWeight &elem1, const TValueWithWeight &elem2)
         { return elem1.Value < elem2.Value; });
    Sort(origs, [](const TValueWithWeight &elem1, const TValueWithWeight &elem2)
         { return elem1.Value < elem2.Value; });

    TVector<double> leftDer;
    TVector<double> rightDer;
    leftDer.yresize(sampleSize);
    rightDer.yresize(sampleSize);
    for (auto i : xrange(sampleSize))
    {
        double a = select_quantile(boundaries, quantiles, origs[i].Weight);
        leftDer[i] = -a * elements[i].Weight;
        rightDer[i] = (1.0 - a) * elements[i].Weight;
    }
    TVector<double> leftDerAccum;
    TVector<double> rightDerAccum;
    leftDerAccum.yresize(sampleSize);
    rightDerAccum.yresize(sampleSize);
    leftDerAccum[sampleSize - 1] = 0;
    rightDerAccum[0] = 0;
    for (auto i : xrange(size_t(1), sampleSize))
    {
        leftDerAccum[sampleSize - 1 - i] = leftDerAccum[sampleSize - i] + leftDer[sampleSize - i];
        rightDerAccum[i] = rightDer[i-1] + rightDerAccum[i - 1];
    }
    size_t l = 0;
    size_t r = sampleSize - 1;
    size_t n = 0;
    for (auto it : xrange(BINARY_SEARCH_ITERATIONS))
    {
        Y_UNUSED(it);
        if (r - l == 1)
        {
            n = fabs(rightDerAccum[l] + leftDerAccum[l]) < fabs(rightDerAccum[r] + leftDerAccum[r]) ? l : r;
            break;
        }
        n = l + ceil((r - l) / 2.);
        if ((n == r) || (fabs(rightDerAccum[n] + leftDerAccum[n]) < 1.e-9))
            break;
        if ((rightDerAccum[n] + leftDerAccum[n]) * (rightDerAccum[l] + leftDerAccum[l]) < 0)
            r = n;
        else
            l = n;
    }
    return elements[n].Value;
}

double CalcTargetDependentMinimum(
    TConstArrayRef<float> sampleRef,
    TConstArrayRef<float> weightsRef,
    TConstArrayRef<float> origTarget,
    TConstArrayRef<double> boundaries,
    TConstArrayRef<double> quantiles)
{
    if (sampleRef.empty())
    {
        return 0.0;
    }
    TVector<float> defaultWeights;
    if (weightsRef.empty())
    {
        defaultWeights.resize(sampleRef.size(), 1.0);
        weightsRef = defaultWeights;
    }
    Y_ASSERT(sampleRef.size() == weightsRef.size());
    return CalcTargetDependentMinimumBinarySearch(sampleRef, weightsRef, origTarget, boundaries, quantiles);
}
