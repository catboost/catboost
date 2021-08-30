#include "precision_recall_at_k.h"
#include "doc_comparator.h"
#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/set.h>
#include <util/generic/vector.h>
#include <util/generic/utility.h>

static size_t CalcSampleSize(size_t approxSize, int top) {
    return (top < 0 || approxSize < static_cast<size_t>(top)) ? approxSize : static_cast<size_t>(top);
}

static TVector<std::pair<double, float>> UnionApproxAndTarget(TConstArrayRef<double> approx,
                                                              TConstArrayRef<float> target) {
    TVector<std::pair<double, float>> pairs;
    pairs.clear();

    for (size_t index = 0; index < approx.size(); ++index) {
        pairs.push_back(std::make_pair(approx[index], target[index]));
    }

    return pairs;
};

static TVector<std::pair<double, float>> GetSortedApproxAndTarget(TConstArrayRef<double> approx,
                                                                  TConstArrayRef<float> target,
                                                                  size_t top) {
    TVector<std::pair<double, float>> approxAndTarget = UnionApproxAndTarget(approx, target);
    std::nth_element(approxAndTarget.begin(), approxAndTarget.begin() + top, approxAndTarget.end(),
                [](const std::pair<double, float>& left, const std::pair<double, float>& right) {
                    return CompareDocs(left.first, left.second, right.first, right.second);
                });
    return approxAndTarget;
};

static int CalcRelevant(TConstArrayRef<std::pair<double, float>> approxAndTarget, float border, size_t size) {
    int relevant = 0;
    for (size_t i = 0; i < size; i++) {
        if (approxAndTarget[i].second > border)
            relevant++;
    }
    return relevant;
}

static int CalcRelevant(TConstArrayRef<std::pair<double, float>> approxAndTarget, float border) {
    return CalcRelevant(approxAndTarget, border, approxAndTarget.size());
}

double CalcPrecisionAtK(TConstArrayRef<double> approx, TConstArrayRef<float> target, int top, float border) {
    size_t size = CalcSampleSize(target.size(), top);
    TVector<std::pair<double, float>> approxAndTarget = GetSortedApproxAndTarget(approx, target, size);
    return CalcRelevant(approxAndTarget, border, size) / static_cast<double>(size);
}

double CalcRecallAtK(TConstArrayRef<double> approx, TConstArrayRef<float> target, int top, float border) {
    size_t size = CalcSampleSize(target.size(), top);
    TVector<std::pair<double, float>> approxAndTarget = GetSortedApproxAndTarget(approx, target, size);
    int relevant = CalcRelevant(approxAndTarget, border);
    return relevant != 0 ? CalcRelevant(approxAndTarget, border, size) / static_cast<double>(relevant) : 1;
}

double CalcAveragePrecisionK(TConstArrayRef<double> approx, TConstArrayRef<float> target, int top, float border) {
    double score = 0;
    double hits = 0;

    size_t size = CalcSampleSize(target.size(), top);
    TVector<std::pair<double, float>> approxAndTarget = UnionApproxAndTarget(approx, target);

    PartialSort(approxAndTarget.begin(), approxAndTarget.begin() + size, approxAndTarget.end(),
                [](const std::pair<double, float>& left, const std::pair<double, float>& right) {
                    return CompareDocs(left.first, left.second, right.first, right.second);
                });

    for (size_t index = 0; index < approxAndTarget.size(); ++index) {
        if (approxAndTarget[index].second > border) {
            hits += 1;
            if (index < size) {
                score += hits / (index + 1);
            }
        }
    }
    return hits > 0 ? score / Min<double>(hits, static_cast<size_t>(size)) : 0;
}
