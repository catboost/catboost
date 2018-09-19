#include "error_functions.h"

#include <util/generic/xrange.h>

void TCrossEntropyError::CalcFirstDerRange(
     int start,
     int count,
     const double* __restrict approxes,
     const double* __restrict approxDeltas,
     const float* __restrict targets,
     const float* __restrict weights,
     double* __restrict ders
) const {
    if (approxDeltas != nullptr) {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double e = approxes[i] * approxDeltas[i];
            const double p = e / (1 + e);
            ders[i] = targets[i] - p;
        }
    } else {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double e = approxes[i];
            const double p = e / (1 + e);
            ders[i] = targets[i] - p;
        }
    }
    if (weights != nullptr) {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            ders[i] *= weights[i];
        }
    }
}

template<bool CalcThirdDer>
static void CalcCrossEntropyErrorDersRangeImpl(
    int start,
    int count,
    const double* __restrict approxExps,
    const double* __restrict approxDeltas,
    const float* __restrict targets,
    const float* __restrict weights,
    TDers* __restrict ders
) {
    if (approxDeltas != nullptr) {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double p = approxExps[i] * approxDeltas[i] / (1 + approxExps[i] * approxDeltas[i]);
            ders[i].Der1 = targets[i] - p;
            ders[i].Der2 = -p * (1 - p);
            if (CalcThirdDer) {
                ders[i].Der3 = -p * (1 - p) * (1 - 2 * p);
            }
        }
    } else {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            const double p = approxExps[i] / (1 + approxExps[i]);
            ders[i].Der1 = targets[i] - p;
            ders[i].Der2 = -p * (1 - p);
            if (CalcThirdDer) {
                ders[i].Der3 = -p * (1 - p) * (1 - 2 * p);
            }
        }
    }
    if (weights != nullptr) {
#pragma clang loop vectorize_width(8) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            ders[i].Der1 *= weights[i];
            ders[i].Der2 *= weights[i];
            if (CalcThirdDer) {
                ders[i].Der3 *= weights[i];
            }
        }
    }
}

void TCrossEntropyError::CalcDersRange(
    int start,
    int count,
    bool calcThirdDer,
    const double* __restrict approxExps,
    const double* __restrict approxDeltas,
    const float* __restrict targets,
    const float* __restrict weights,
    TDers* __restrict ders
) const {
    if (calcThirdDer) {
        CalcCrossEntropyErrorDersRangeImpl<true>(start, count, approxExps, approxDeltas, targets, weights, ders);
    } else {
        CalcCrossEntropyErrorDersRangeImpl<false>(start, count, approxExps, approxDeltas, targets, weights, ders);
    }
}

namespace {
    template<int Capacity>
    class TExpForwardView {
    public:
        TExpForwardView(TConstArrayRef<double> src, double bias)
        : Src(src)
        , Bias(bias)
        {
            ExpSrc.fill(0.0);
        }
        double operator[](size_t idx) {
            Y_ASSERT(ViewBegin <= idx);
            if (ViewEnd <= idx) {
                ViewBegin = idx;
                ViewEnd = Min(idx + Capacity, Src.size());
                for (size_t idx : xrange(ViewBegin, ViewEnd)) {
                    ExpSrc[idx - ViewBegin] = Src[idx] + Bias;
                }
                FastExpInplace(ExpSrc.data(), ViewEnd - ViewBegin);
            }
            return ExpSrc[idx - ViewBegin];
        }
    private:
        TConstArrayRef<double> Src;
        double Bias;
        size_t ViewBegin = 0;
        size_t ViewEnd = 0;
        std::array<double, Capacity> ExpSrc;
    };
}

void TQuerySoftMaxError::CalcDersForSingleQuery(
    int start,
    int offset,
    int count,
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> targets,
    TConstArrayRef<float> weights,
    TArrayRef<TDers> ders
) const {
    double maxApprox = -std::numeric_limits<double>::max();
    float sumWeightedTargets = 0;
    for (int dim = offset; dim < offset + count; ++dim) {
        const float weight = weights.empty() ? 1.0f : weights[start + dim];
        if (weight > 0) {
            maxApprox = std::max(maxApprox, approxes[start + dim]);
            if (targets[start + dim] > 0) {
                sumWeightedTargets += weight * targets[start + dim];
            }
        }
    }
    if (sumWeightedTargets > 0) {
        TExpForwardView</*Capacity*/16> expApproxes(MakeArrayRef(approxes.data(), offset + count), -maxApprox);
        double sumExpApprox = 0;
        for (int dim = offset; dim < offset + count; ++dim) {
            const float weight = weights.empty() ? 1.0f : weights[start + dim];
            if (weight > 0) {
                const double expApprox = expApproxes[start + dim] * weight;
                ders[dim].Der1 = expApprox;
                sumExpApprox += expApprox;
            }
        }
        for (int dim = offset; dim < offset + count; ++dim) {
            const float weight = weights.empty() ? 1.0f : weights[start + dim];
            if (weight > 0) {
                const double p = ders[dim].Der1 / sumExpApprox;
                ders[dim].Der2 = sumWeightedTargets * (p * (p - 1.0) - LambdaReg);
                ders[dim].Der1 = -sumWeightedTargets * p;
                if (targets[start + dim] > 0) {
                    ders[dim].Der1 += weight * targets[start + dim];
                }
            } else {
                ders[dim].Der2 = 0.0;
                ders[dim].Der1 = 0.0;
            }
        }
    } else {
        for (int dim = offset; dim < offset + count; ++dim) {
            ders[dim].Der2 = 0.0;
            ders[dim].Der1 = 0.0;
        }
    }
}

void CheckDerivativeOrderForTrain(ui32 derivativeOrder, ELeavesEstimation estimationMethod) {
    if (estimationMethod == ELeavesEstimation::Newton) {
        CB_ENSURE(derivativeOrder >= 2, "Current error function doesn't support Newton leaves estimation method");
    }
}

void CheckDerivativeOrderForObjectImportance(ui32 derivativeOrder, ELeavesEstimation estimationMethod) {
    CB_ENSURE(derivativeOrder >= 2, "Current error function doesn't support object importance calculation");
    if (estimationMethod == ELeavesEstimation::Newton) {
        CB_ENSURE(derivativeOrder >= 3, "Current error function doesn't support object importance calculation with Newton leaves estimation method");
    }
}
