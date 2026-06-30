#include "dcg.h"
#include "doc_comparator.h"
#include "sample.h"

#include <catboost/libs/helpers/exception.h>

#include <library/cpp/containers/stack_vector/stack_vec.h>
#include <library/cpp/dot_product/dot_product.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>

using NMetrics::TSample;

template <typename F>
static TStackVec<double> GetTopSortedTargets(
    const TConstArrayRef<TSample> samples,
    const ui32 topSizeRequested,
    F&& cmp
) {
    const ui32 topSize = Min<ui32>(topSizeRequested, samples.size());

    TStackVec<ui32> indices;
    indices.yresize(samples.size());
    Iota(indices.begin(), indices.end(), static_cast<ui32>(0));

    if (topSizeRequested < samples.size()) {
        PartialSort(
            indices.begin(), indices.begin() + topSize, indices.end(),
            [samples, cmp](const auto lhs, const auto rhs) {
                if (cmp(samples[lhs], samples[rhs])) {
                    return true;
                }
                if (cmp(samples[rhs], samples[lhs])) {
                    return false;
                }
                return lhs < rhs;
        });
    } else {
        StableSort(
            indices.begin(), indices.end(),
            [samples, cmp](const auto lhs, const auto rhs) {
                return cmp(samples[lhs], samples[rhs]);
        });
    }

    Y_ASSERT(samples.size() == indices.size());
    TStackVec<double> targets;
    targets.yresize(topSize);
    for (size_t i = 0, iEnd = topSize; i < iEnd; ++i) {
        targets[i] = samples[indices[i]].Target;
    }

    return targets;
}

double CalcDcgSorted(
        const TConstArrayRef<double> sortedTargets,
        TConstArrayRef<double> decay,
        const ENdcgMetricType type
) {
    const auto size = sortedTargets.size();

    TStackVec<double> modifiedTargetsHolder;
    TConstArrayRef<double> modifiedTargets = sortedTargets;
    if (ENdcgMetricType::Exp == type) {
        modifiedTargetsHolder.yresize(size);
        for (size_t i = 0; i < size; ++i) {
            modifiedTargetsHolder[i] = pow(2, sortedTargets[i]) - 1.;
        }
        modifiedTargets = modifiedTargetsHolder;
    }

    return DotProduct(modifiedTargets.data(), decay.data(), size);
}

void FillDcgDecay(ENdcgDenominatorType denominator, TMaybe<double> expDecay, TArrayRef<double> decay) {
    decay[0] = 1.;
    if (expDecay.Defined()) {
        for (size_t i = 1; i < decay.size(); ++i) {
            decay[i] = decay[i - 1] * *expDecay;
        }
        return;
    }
    switch (denominator) {
        case ENdcgDenominatorType::Position: {
            for (size_t i = 1; i < decay.size(); ++i) {
                decay[i] = 1. / (i + 1);
            }
            break;
        }
        case ENdcgDenominatorType::LogPosition: {
            for (size_t i = 1; i < decay.size(); ++i) {
                decay[i] = 1. / Log2(static_cast<double>(i + 2));
            }
            break;
        }
        default: {
            CB_ENSURE(false, "Unexpected NDCG denominator type");
        }
    }

}

double CalcDcg(TConstArrayRef<TSample> samples, TConstArrayRef<double> decay, ENdcgMetricType type, ui32 topSize) {
    const auto sortedTargets = GetTopSortedTargets(samples, topSize, [](const auto& left, const auto& right) {
        return CompareDocs(left.Prediction, left.Target, right.Prediction, right.Target);
    });
    return CalcDcgSorted(sortedTargets, decay, type);
}

double CalcDcg(TConstArrayRef<TSample> samples, ENdcgMetricType type, TMaybe<double> expDecay, ui32 topSize,
               ENdcgDenominatorType denominator) {
    const auto sortedTargets = GetTopSortedTargets(samples, topSize, [](const auto& left, const auto& right) {
        return CompareDocs(left.Prediction, left.Target, right.Prediction, right.Target);
    });
    TStackVec<double> decay;
    decay.yresize(sortedTargets.size());
    FillDcgDecay(denominator, expDecay, decay);
    return CalcDcgSorted(sortedTargets, decay, type);
}

double CalcIDcg(TConstArrayRef<TSample> samples, TConstArrayRef<double> decay, ENdcgMetricType type, ui32 topSize) {
    const auto sortedTargets = GetTopSortedTargets(samples, topSize, [](const auto& left, const auto& right) {
        return left.Target > right.Target;
    });
    return CalcDcgSorted(sortedTargets, decay, type);
}

double CalcIDcg(TConstArrayRef<TSample> samples, ENdcgMetricType type, TMaybe<double> expDecay, ui32 topSize,
                ENdcgDenominatorType denominator) {
    const auto sortedTargets = GetTopSortedTargets(samples, topSize, [](const auto& left, const auto& right) {
        return left.Target > right.Target;
    });
    TStackVec<double> decay;
    decay.yresize(sortedTargets.size());
    FillDcgDecay(denominator, expDecay, decay);
    return CalcDcgSorted(sortedTargets, decay, type);
}

double CalcNdcg(TConstArrayRef<TSample> samples, TConstArrayRef<double> decay, ENdcgMetricType type, ui32 topSize) {
    double dcg = CalcDcg(samples, decay, type, topSize);
    double idcg = CalcIDcg(samples, decay, type, topSize);
    return idcg > 0 ? dcg / idcg : 1;
}
