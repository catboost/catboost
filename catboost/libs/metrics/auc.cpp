#include "auc.h"

#include <util/generic/algorithm.h>

using NMetrics::TSample;

static double MergeAndCountInversions(TVector<TSample>* samples, TVector<TSample>* aux, ui32 lo, ui32 hi, ui32 mid) {
    double result = 0;
    ui32 left = lo;
    ui32 right = mid;
    auto& input = *samples;
    auto& output = *aux;
    ui32 outputIndex = lo;
    double accumulatedWeight = 0;
    while (outputIndex < hi) {
        if (left == mid || right < hi && input[right].Target < input[left].Target) {
            accumulatedWeight += input[right].Weight;
            output[outputIndex] = input[right];
            ++outputIndex;
            ++right;
        } else {
            result += input[left].Weight * accumulatedWeight;
            output[outputIndex] = input[left];
            ++outputIndex;
            ++left;
        }
    }
    return result;
}

static double SortAndCountInversions(TVector<TSample>* samples, TVector<TSample>* aux, ui32 lo, ui32 hi) {
    if (lo + 1 >= hi) return 0;
    ui32 mid = lo + (hi - lo) / 2;
    auto leftCount = SortAndCountInversions(samples, aux, lo, mid);
    auto rightCount = SortAndCountInversions(samples, aux, mid, hi);
    auto mergeCount = MergeAndCountInversions(samples, aux, lo, hi, mid);
    std::copy(aux->begin() + lo, aux->begin() + hi, samples->begin() + lo);
    return leftCount + rightCount + mergeCount;
}

double CalcAUC(TVector<TSample>* samples, double* outWeightSum, double* outPairWeightSum) {
    double weightSum = 0;
    double pairWeightSum = 0;
    Sort(samples->begin(), samples->end(), [](const TSample& left, const TSample& right) {
        return left.Target < right.Target;
    });
    double accumulatedWeight = 0;
    for (ui32 i = 0; i < samples->size(); ++i) {
        auto& sample = (*samples)[i];
        if (i > 0 && (*samples)[i - 1].Target != sample.Target) {
            accumulatedWeight = weightSum;
        }
        weightSum += sample.Weight;
        pairWeightSum += accumulatedWeight * sample.Weight;
    }
    if (outWeightSum != nullptr) {
        *outWeightSum = weightSum;
    }
    if (outPairWeightSum != nullptr) {
        *outPairWeightSum = pairWeightSum;
    }
    if (pairWeightSum == 0) {
        return 0;
    }
    TVector<TSample> aux(samples->begin(), samples->end());
    Sort(samples->begin(), samples->end(), [](const TSample& left, const TSample& right) {
        return left.Prediction < right.Prediction ||
               left.Prediction == right.Prediction && left.Target < right.Target;
    });
    auto optimisticAUC = 1 - SortAndCountInversions(samples, &aux, 0, samples->size()) / pairWeightSum;
    Sort(samples->begin(), samples->end(), [](const TSample& left, const TSample& right) {
        return left.Prediction < right.Prediction ||
               left.Prediction == right.Prediction && left.Target > right.Target;
    });
    auto pessimisticAUC = 1 - SortAndCountInversions(samples, &aux, 0, samples->size()) / pairWeightSum;
    return (optimisticAUC + pessimisticAUC) / 2.0;
}

