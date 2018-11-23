#include "dcg.h"
#include "doc_comparator.h"
#include "sample.h"

#include <library/dot_product/dot_product.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>

using NMetrics::TSample;

template <typename F>
static TVector<double> GetSortedTargets(const TConstArrayRef<TSample> samples, F&& cmp) {
    TVector<ui32> indices;
    indices.yresize(samples.size());
    Iota(indices.begin(), indices.end(), static_cast<ui32>(0));

    Sort(indices, [samples, cmp](const auto lhs, const auto rhs) {
        return cmp(samples[lhs], samples[rhs]);
    });

    Y_ASSERT(samples.size() == indices.size());
    TVector<double> targets;
    targets.yresize(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        targets[i] = samples[indices[i]].Target;
    }

    return targets;
}

static double CalcDcgSorted(
    const TConstArrayRef<double> sortedTargets,
    const ENdcgMetricType type,
    const TMaybe<double> expDecay)
{
    const auto size = sortedTargets.size();

    TVector<double> decay;
    decay.yresize(size);
    decay.front() = 1.;
    if (expDecay.Defined()) {
        const auto expDecayBase = *expDecay;
        for (size_t i = 1; i < size; ++i) {
            decay[i] = decay[i - 1] * expDecayBase;
        }
    } else {
        for (size_t i = 1; i < size; ++i) {
            decay[i] = 1. / Log2(static_cast<double>(i + 2));
        }
    }

    TVector<double> modifiedTargetsHolder;
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

double CalcDcg(TConstArrayRef<TSample> samples, ENdcgMetricType type, TMaybe<double> expDecay) {
    const auto sortedTargets = GetSortedTargets(samples,  [](const auto& left, const auto& right) {
        return CompareDocs(left.Prediction, left.Target, right.Prediction, right.Target);
    });
    return CalcDcgSorted(sortedTargets, type, expDecay);
}

double CalcIDcg(TConstArrayRef<TSample> samples, ENdcgMetricType type, TMaybe<double> expDecay) {
    const auto sortedTargets = GetSortedTargets(samples, [](const auto& left, const auto& right) {
        return left.Target > right.Target;
    });
    return CalcDcgSorted(sortedTargets, type, expDecay);
}

double CalcNdcg(TConstArrayRef<TSample> samples, ENdcgMetricType type) {
    double dcg = CalcDcg(samples, type);
    double idcg = CalcIDcg(samples, type);
    return idcg > 0 ? dcg / idcg : 0;
}
