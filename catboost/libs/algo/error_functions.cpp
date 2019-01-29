#include "error_functions.h"

#include <util/generic/xrange.h>

template <int MaxDerivativeOrder, bool UseTDers, bool UseExpApprox, bool HasDelta>
void IDerCalcer::CalcDersRangeImpl(
    int start,
    int count,
    const double* approxes,
    const double* approxDeltas,
    const float* targets,
    const float* weights,
    TDers* ders,
    double* firstDers
) const {
    Y_ASSERT(UseExpApprox == IsExpApprox);
    Y_ASSERT(HasDelta == (approxDeltas != nullptr));
    Y_ASSERT(UseTDers == (ders != nullptr) && (ders != nullptr) == (firstDers == nullptr));
    Y_ASSERT(MaxDerivativeOrder <= MaxSupportedDerivativeOrder);
    Y_ASSERT((MaxDerivativeOrder > 1) <= (ders != nullptr));
    for (int i = start; i < start + count; ++i) {
        double updatedApprox = approxes[i];
        if (HasDelta) {
            updatedApprox = UpdateApprox<UseExpApprox>(updatedApprox, approxDeltas[i]);
        }
        if (UseTDers) {
            ders[i].Der1 = CalcDer(updatedApprox, targets[i]);
        } else {
            firstDers[i] = CalcDer(updatedApprox, targets[i]);
        }
        if (MaxDerivativeOrder >= 2) {
            ders[i].Der2 = CalcDer2(updatedApprox, targets[i]);
        }
        if (MaxDerivativeOrder >= 3) {
            ders[i].Der3 = CalcDer3(updatedApprox, targets[i]);
        }
    }
    if (weights != nullptr) {
        for (int i = start; i < start + count; ++i) {
            if (UseTDers) {
                ders[i].Der1 *= weights[i];
            } else {
                firstDers[i] *= weights[i];
            }
            if (MaxDerivativeOrder >= 2) {
                ders[i].Der2 *= weights[i];
            }
            if (MaxDerivativeOrder >= 3) {
                ders[i].Der3 *= weights[i];
            }
        }
    }
}

static constexpr int EncodeImplParameters(int maxDerivativeOrder, bool useTDers, bool isExpApprox, bool hasDelta) {
    return maxDerivativeOrder * 8 + useTDers * 4 + isExpApprox * 2 + hasDelta;
}

void IDerCalcer::CalcDersRange(
    int start,
    int count,
    int maxDerivativeOrder,
    const double* approxes,
    const double* approxDeltas,
    const float* targets,
    const float* weights,
    TDers* ders,
    double* firstDers
) const {
    const bool hasDelta = approxDeltas != nullptr;
    const bool useTDers = ders != nullptr;
    switch (EncodeImplParameters(maxDerivativeOrder, useTDers, IsExpApprox, hasDelta)) {
        case EncodeImplParameters(1, false, false, false):
            return CalcDersRangeImpl<1, false, false, false>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(1, false, false, true):
            return CalcDersRangeImpl<1, false, false, true>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(1, false, true, false):
            return CalcDersRangeImpl<1, false, true, false>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(1, false, true, true):
            return CalcDersRangeImpl<1, false, true, true>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(1, true, false, false):
            return CalcDersRangeImpl<1, true, false, false>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(1, true, false, true):
            return CalcDersRangeImpl<1, true, false, true>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(1, true, true, false):
            return CalcDersRangeImpl<1, true, true, false>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(1, true, true, true):
            return CalcDersRangeImpl<1, true, true, true>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(2, true, false, false):
            return CalcDersRangeImpl<2, true, false, false>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(2, true, false, true):
            return CalcDersRangeImpl<2, true, false, true>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(2, true, true, false):
            return CalcDersRangeImpl<2, true, true, false>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(2, true, true, true):
            return CalcDersRangeImpl<2, true, true, true>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(3, true, false, false):
            return CalcDersRangeImpl<3, true, false, false>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(3, true, false, true):
            return CalcDersRangeImpl<3, true, false, true>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(3, true, true, false):
            return CalcDersRangeImpl<3, true, true, false>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        case EncodeImplParameters(3, true, true, true):
            return CalcDersRangeImpl<3, true, true, true>(start, count, approxes, approxDeltas, targets, weights, ders, firstDers);
        default:
            Y_ASSERT(false);
    }
}

namespace {
    template <int Capacity>
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

template <bool CalcThirdDer, bool UseTDers, bool UseExpApprox, bool HasDelta>
static void CalcCrossEntropyDerRangeImpl(
    int start,
    int count,
    const double* approxes,
    const double* approxDeltas,
    const float* targets,
    const float* weights,
    TDers* ders,
    double* firstDers
) {
    TExpForwardView</*Capacity*/16> expApproxes(MakeArrayRef(approxes + start, count), 0);
    TExpForwardView</*Capacity*/16> expApproxDeltas(MakeArrayRef(approxDeltas + start, count), 0);
    Y_ASSERT(HasDelta == (approxDeltas != nullptr));
#pragma clang loop vectorize_width(4) interleave_count(2)
    for (int i = start; i < start + count; ++i) {
        double e;
        if (UseExpApprox) {
            e = approxes[i];
        } else {
            e = expApproxes[i - start];
        }
        if (HasDelta) {
            if (UseExpApprox) {
                e *= approxDeltas[i];
            } else {
                e *= expApproxDeltas[i - start];
            }
        }
        const double p = 1 - 1 / (1 + e);
        if (UseTDers) {
            ders[i].Der1 = targets[i] - p;
            ders[i].Der2 = -p * (1 - p);
            if (CalcThirdDer) {
                ders[i].Der3 = -p * (1 - p) * (1 - 2 * p);
            }
        } else {
            firstDers[i] = targets[i] - p;
        }
    }
    if (weights != nullptr) {
#pragma clang loop vectorize_width(4) interleave_count(2)
        for (int i = start; i < start + count; ++i) {
            if (UseTDers) {
                ders[i].Der1 *= weights[i];
                ders[i].Der2 *= weights[i];
                if (CalcThirdDer) {
                    ders[i].Der3 *= weights[i];
                }
            } else {
                firstDers[i] *= weights[i];
            }
        }
    }
}

static constexpr int EncodeCrossEntropyImplParameters(bool CalcThirdDer, bool UseTDers, bool UseExpApprox, bool HasDelta) {
    return CalcThirdDer * 8 + UseTDers * 4 + UseExpApprox * 2 + HasDelta;
}

void TCrossEntropyError::CalcFirstDerRange(
    int start,
    int count,
    const double* approxes,
    const double* approxDeltas,
    const float* targets,
    const float* weights,
    double* ders
) const {
    const int encodedParameters = EncodeCrossEntropyImplParameters(/*CalcThirdDer*/false, /*UseTDers*/false, GetIsExpApprox(), /*HasDelta*/ approxDeltas != nullptr);
    switch (encodedParameters) {
        case EncodeCrossEntropyImplParameters(false, false, true, true):
            return CalcCrossEntropyDerRangeImpl<false, false, true, true>(start, count, approxes, approxDeltas, targets, weights, nullptr, ders);
        case EncodeCrossEntropyImplParameters(false, false, true, false):
            return CalcCrossEntropyDerRangeImpl<false, false, true, false>(start, count, approxes, approxDeltas, targets, weights, nullptr, ders);
        case EncodeCrossEntropyImplParameters(false, false, false, true):
            return CalcCrossEntropyDerRangeImpl<false, false, false, true>(start, count, approxes, approxDeltas, targets, weights, nullptr, ders);
        case EncodeCrossEntropyImplParameters(false, false, false, false):
            return CalcCrossEntropyDerRangeImpl<false, false, false, false>(start, count, approxes, approxDeltas, targets, weights, nullptr, ders);
        default:
            Y_ASSERT(false);
    }
}

void TCrossEntropyError::CalcDersRange(
    int start,
    int count,
    bool calcThirdDer,
    const double* approxes,
    const double* approxDeltas,
    const float* targets,
    const float* weights,
    TDers* ders
) const {
    const int encodedParameters = EncodeCrossEntropyImplParameters(calcThirdDer, /*UseTDers*/true, GetIsExpApprox(), /*HasDelta*/ approxDeltas != nullptr);
    switch (encodedParameters) {
        case EncodeCrossEntropyImplParameters(true, true, true, true):
            return CalcCrossEntropyDerRangeImpl<true, true, true, true>(start, count, approxes, approxDeltas, targets, weights, ders, nullptr);
        case EncodeCrossEntropyImplParameters(true, true, true, false):
            return CalcCrossEntropyDerRangeImpl<true, true, true, false>(start, count, approxes, approxDeltas, targets, weights, ders, nullptr);
        case EncodeCrossEntropyImplParameters(true, true, false, true):
            return CalcCrossEntropyDerRangeImpl<true, true, false, true>(start, count, approxes, approxDeltas, targets, weights, ders, nullptr);
        case EncodeCrossEntropyImplParameters(true, true, false, false):
            return CalcCrossEntropyDerRangeImpl<true, true, false, false>(start, count, approxes, approxDeltas, targets, weights, ders, nullptr);
        case EncodeCrossEntropyImplParameters(false, true, true, true):
            return CalcCrossEntropyDerRangeImpl<false, true, true, true>(start, count, approxes, approxDeltas, targets, weights, ders, nullptr);
        case EncodeCrossEntropyImplParameters(false, true, true, false):
            return CalcCrossEntropyDerRangeImpl<false, true, true, false>(start, count, approxes, approxDeltas, targets, weights, ders, nullptr);
        case EncodeCrossEntropyImplParameters(false, true, false, true):
            return CalcCrossEntropyDerRangeImpl<false, true, false, true>(start, count, approxes, approxDeltas, targets, weights, ders, nullptr);
        case EncodeCrossEntropyImplParameters(false, true, false, false):
            return CalcCrossEntropyDerRangeImpl<false, true, false, false>(start, count, approxes, approxDeltas, targets, weights, ders, nullptr);
        default:
            Y_ASSERT(false);
    }
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
