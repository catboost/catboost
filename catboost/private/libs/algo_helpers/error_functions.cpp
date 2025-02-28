#include "error_functions.h"

#include <catboost/libs/helpers/dispatch_generic_lambda.h>
#include <catboost/libs/metrics/dcg.h>

#include <util/generic/xrange.h>
#include <util/random/normal.h>


using namespace NCB;


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
    DispatchGenericLambda(
        [=] (auto useTDers, auto isExpApprox, auto hasDelta) {
            switch (maxDerivativeOrder) {
                case 1:
                    return CalcDersRangeImpl<1, useTDers, isExpApprox, hasDelta>(
                        start,
                        count,
                        approxes,
                        approxDeltas,
                        targets,
                        weights,
                        ders,
                        firstDers);
                case 2:
                    return CalcDersRangeImpl<2, useTDers, isExpApprox, hasDelta>(
                        start,
                        count,
                        approxes,
                        approxDeltas,
                        targets,
                        weights,
                        ders,
                        firstDers);
                case 3:
                    return CalcDersRangeImpl<3, useTDers, isExpApprox, hasDelta>(
                        start,
                        count,
                        approxes,
                        approxDeltas,
                        targets,
                        weights,
                        ders,
                        firstDers);
                default:
                    CB_ENSURE(false, "Only 1st, 2nd, and 3rd derivatives are supported");
            }
    },
    useTDers, IsExpApprox, hasDelta);
}

static TVector<int> ArgSort(
    int start,
    int count,
    const float* targets
) {
    TVector<int> labelOrder(count);
    std::iota(labelOrder.begin(), labelOrder.end(), start);
    StableSort(
        labelOrder.begin(),
        labelOrder.end(),
        [=] (int lhs, int rhs) {
            return std::abs(targets[lhs]) < std::abs(targets[rhs]);
        }
    );

    return labelOrder;
}

static double CalcCoxApproxSum(
    int start,
    int count,
    double maxApprox,
    const double* approxes,
    const double* approxesDeltas
) {
    double expPSum = 0;
    for (auto i : xrange(start, start + count)) {
        double updatedApprox = approxes[i];
        if (approxesDeltas != nullptr) {
            updatedApprox += approxesDeltas[i];
        }
        expPSum += std::exp(updatedApprox - maxApprox);
    }

    return expPSum;
}

void TCoxError::CalcDersRange(
    int start,
    int count,
    bool /*calcThirdDer*/,
    const double* approxes,
    const double* approxesDeltas,
    const float* targets,
    const float* /*weights*/,
    TDers* ders
) const {
    const TVector<int> labelOrder = ArgSort(start, count, targets);

    const auto getApprox = [=] (int i) {
        return approxes[i] + (approxesDeltas == nullptr ? 0 : approxesDeltas[i]);
    };

    double maxApprox = getApprox(start);
    for (auto i : xrange(start + 1, start + count)) {
        const auto approx = getApprox(i);
        if (approx > maxApprox) {
            maxApprox = approx;
        }
    }

    double expPSum = CalcCoxApproxSum(start, count, maxApprox, approxes, approxesDeltas);

    double rk = 0;
    double sk = 0;
    double lastExpP = 0.0;
    double accumulatedSum = 0;
    for (auto i : xrange(start, start + count)) {
        const int ind = labelOrder[i];
        const double p = getApprox(ind) - maxApprox;

        const double expP = std::exp(p);
        const double y = targets[ind];
        accumulatedSum += lastExpP;

        if (y > 0) {
            expPSum -= accumulatedSum;
            accumulatedSum = 0;
            rk += 1.0 / expPSum;
            sk += 1.0 / (expPSum * expPSum);
        }

        const double grad = static_cast<double>(y > 0) - expP * rk;
        const double hess = expP * rk - expP * expP * sk;
        ders[ind].Der1 = grad;
        ders[ind].Der2 = - hess;

        lastExpP = expP;
    }
}


void TCoxError::CalcFirstDerRange(
    int start,
    int count,
    const double* approxes,
    const double* approxesDeltas,
    const float* targets,
    const float* /*weights*/,
    double* firstDers
) const {
    const TVector<int> labelOrder = ArgSort(start, count, targets);

    const auto getApprox = [=] (int i) {
        return approxes[i] + (approxesDeltas == nullptr ? 0 : approxesDeltas[i]);
    };

    double maxApprox = getApprox(0);
    for (auto i : xrange(start, start + count)) {
        const auto approx = getApprox(i);
        if (approx > maxApprox) {
            maxApprox = approx;
        }
    }

    double expPSum = CalcCoxApproxSum(start, count, maxApprox, approxes, approxesDeltas);

    double rk = 0;
    double lastExpP = 0.0;
    double accumulatedSum = 0;
    for (auto i : xrange(start, start + count)) {
        const int ind = labelOrder[i];
        const double p = getApprox(ind) - maxApprox;

        const double expP = std::exp(p);
        const double y = targets[ind];
        accumulatedSum += lastExpP;

        if (y > 0) {
            expPSum -= accumulatedSum;
            accumulatedSum = 0;
            rk += 1.0 / expPSum;
        }

        const double grad = static_cast<double>(y > 0) - expP * rk;
        firstDers[ind] = grad;

        lastExpP = expP;
    }
}

namespace {
    template <int Capacity>
    class TExpForwardView {
    public:
        explicit TExpForwardView(TConstArrayRef<double> src)
            : Src(src)
        {
            ExpSrc.fill(0.0);
        }
        TExpForwardView(TConstArrayRef<double> src, double bias, double scale)
            : Src(src)
            , Bias(bias)
            , Scale(scale)
            , IsWithBiasScale(bias != 0.0 || scale != 1.0)
        {
            ExpSrc.fill(0.0);
        }
        double operator[](size_t idx) {
            Y_ASSERT(ViewBegin <= idx);
            if (ViewEnd <= idx) {
                ViewBegin = idx;
                ViewEnd = Min(idx + Capacity, Src.size());
                if (IsWithBiasScale) {
                    for (size_t i : xrange(ViewBegin, ViewEnd)) {
                        ExpSrc[i - ViewBegin] = (Src[i] + Bias) * Scale;
                    }
                } else {
                    for (size_t i : xrange(ViewBegin, ViewEnd)) {
                        ExpSrc[i - ViewBegin] = Src[i];
                    }
                }
                NCB::FastExpWithInfInplace(ExpSrc.data(), ViewEnd - ViewBegin);
            }
            return ExpSrc[idx - ViewBegin];
        }
    private:
        TConstArrayRef<double> Src;
        double Bias = 0.0;
        double Scale = 1.0;
        size_t ViewBegin = 0;
        size_t ViewEnd = 0;
        bool IsWithBiasScale = false;
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
    TExpForwardView</*Capacity*/16> expApproxes(MakeArrayRef(approxes + start, count));
    TExpForwardView</*Capacity*/16> expApproxDeltas(MakeArrayRef(approxDeltas + start, count));
    Y_ASSERT(HasDelta == (approxDeltas != nullptr));
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
#if defined(__clang__) && !defined(CLANG_COVERAGE) && !defined(undefined_sanitizer_enabled)
#pragma clang loop vectorize_width(4) interleave_count(2)
#endif
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

void TSurvivalAftError::CalcDers(
    TConstArrayRef<double> approx,
    TConstArrayRef<float> target,
    float /*weight*/,
    TVector<double>* der,
    THessianInfo* der2
) const {
    double transformedTargetLower = 0, transformedTargetUpper = 0;
    double firstDerNumerator, firstDerDenominator, secondDerNumerator, secondDerDenominator;
    bool targetSign;
    ECensoredType censorType;
    if (target[0] == target[1]) {
        transformedTargetLower = InverseMonotoneTransform(approx[0], target[0], Scale);
        targetSign = transformedTargetLower > 0;
        censorType = ECensoredType::Uncensored;
        const auto pdf = Distribution->CalcPdf(transformedTargetLower);
        const auto der1 = Distribution->CalcPdfDer1(pdf, transformedTargetLower);

        firstDerNumerator = der1;
        firstDerDenominator = Scale * pdf;
        if (der2 != nullptr) {
            Y_ASSERT(der2->HessianType == EHessianType::Diagonal &&
                        der2->ApproxDimension == approx.ysize());

            secondDerNumerator = -(pdf * Distribution->CalcPdfDer2(pdf, transformedTargetLower) - Sqr(der1));
            secondDerDenominator = Sqr(Scale * pdf);
        }
    } else {
        censorType = ECensoredType::IntervalCensored;
        double pdfUpper, pdfLower, cdfUpper, cdfLower, der1Upper, der1Lower;
        if (target[1] == -1) {
            pdfUpper = 0;
            cdfUpper = 1;
            der1Upper = 0;
            censorType = ECensoredType::RightCensored;
        } else {
            transformedTargetUpper = InverseMonotoneTransform(approx[0], target[1], Scale);
            pdfUpper = Distribution->CalcPdf(transformedTargetUpper);
            cdfUpper = Distribution->CalcCdf(transformedTargetUpper);
            der1Upper = Distribution->CalcPdfDer1(pdfUpper, transformedTargetUpper);
        }
        if (target[0] == -1) {
            pdfLower = 0;
            cdfLower = 0;
            der1Lower = 0;
            censorType = ECensoredType::LeftCensored;
        } else {
            transformedTargetLower = InverseMonotoneTransform(approx[0], target[0], Scale);
            pdfLower = Distribution->CalcPdf(transformedTargetLower);
            cdfLower = Distribution->CalcCdf(transformedTargetLower);
            der1Lower = Distribution->CalcPdfDer1(pdfLower, transformedTargetLower);
        }
        targetSign = (transformedTargetLower > 0 || transformedTargetUpper > 0);
        const auto pdfDiff = pdfUpper - pdfLower;
        const auto der1Diff = der1Upper - der1Lower;
        const auto cdfDiff = cdfUpper - cdfLower;

        firstDerNumerator = pdfDiff;
        firstDerDenominator = Scale * cdfDiff;

        if (der2 != nullptr) {
            Y_ASSERT(der2->HessianType == EHessianType::Diagonal &&
                        der2->ApproxDimension == approx.ysize());

            secondDerNumerator = -cdfDiff * der1Diff + Sqr(pdfDiff);
            secondDerDenominator = Sqr(Scale * cdfDiff);
        }
    }


    const auto distributionType = Distribution->GetDistributionType();
    (*der)[0] = firstDerNumerator / firstDerDenominator;
    if (firstDerDenominator < TDerivativeConstants::Epsilon && (IsNan((*der)[0]) || !IsFinite((*der)[0]))) {
        const auto& ders =  DispatchDerivativeLimits(distributionType, EDerivativeOrder::First, censorType, Scale);
        auto minDer1 = std::get<0>(ders);
        auto maxDer1 = std::get<1>(ders);
        (*der)[0] = targetSign ? minDer1 : maxDer1;
    }
    (*der)[0] = -ClipDerivatives((*der)[0], TDerivativeConstants::MinFirstDer, TDerivativeConstants::MaxFirstDer);

    if (der2 != nullptr) {
        der2->Data[0] = secondDerNumerator / secondDerDenominator;
        if (secondDerDenominator < TDerivativeConstants::Epsilon && (IsNan(der2->Data[0]) || !IsFinite(der2->Data[0]))) {
            const auto& ders =  DispatchDerivativeLimits(distributionType, EDerivativeOrder::Second, censorType, Scale);
            auto minDer2 = std::get<0>(ders);
            auto maxDer2 = std::get<1>(ders);
            der2->Data[0] = targetSign ? minDer2 : maxDer2;
        }
        der2->Data[0] = -ClipDerivatives(der2->Data[0], TDerivativeConstants::MinSecondDer, TDerivativeConstants::MaxSecondDer);
    }
}


void TMultiQuantileError::CalcDersMulti(
    const TVector<double>& approx,
    float target,
    float weight,
    TVector<double>* der,
    THessianInfo* der2
) const {
    const auto quantileCount = Alpha.size();
    Y_ASSERT(approx.size() == quantileCount);
    Y_ASSERT(der && der->size() == quantileCount);
    for (auto idx : xrange(quantileCount)) {
        const auto arg = target - approx[idx];
        if (abs(arg) < Delta) {
            (*der)[idx] = 0;
        } else {
            (*der)[idx] = weight * (arg > 0 ? Alpha[idx] : -(1.0 - Alpha[idx]));
        }
    }

    if (der2 != nullptr) {
        Y_ASSERT(
            der2->HessianType == EHessianType::Diagonal
            && der2->ApproxDimension == SafeIntegerCast<int>(quantileCount));
        Fill(der2->Data.begin(), der2->Data.end(), weight * QUANTILE_DER2_AND_DER3);
    }
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
    DispatchGenericLambda(
        [=] (auto useExpApprox, auto hasDelta) {
            CalcCrossEntropyDerRangeImpl<false, false, useExpApprox, hasDelta>(
                start,
                count,
                approxes,
                approxDeltas,
                targets,
                weights,
                nullptr,
                ders);
        },
        GetIsExpApprox(), approxDeltas != nullptr);
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
    DispatchGenericLambda(
        [=] (auto calcThirdDer, auto useExpApprox, auto hasDelta) {
            CalcCrossEntropyDerRangeImpl<calcThirdDer, true, useExpApprox, hasDelta>(
                start,
                count,
                approxes,
                approxDeltas,
                targets,
                weights,
                ders,
                nullptr);
        },
        calcThirdDer, GetIsExpApprox(), approxDeltas != nullptr);
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
        TExpForwardView</*Capacity*/16> expApproxes(MakeArrayRef(approxes.data(), offset + count), -maxApprox, Beta);
        double sumExpApprox = 0;
        for (int dim = offset; dim < offset + count; ++dim) {
            const float weight = weights.empty() ? 1.0f : weights[start + dim];
            const double expApprox = expApproxes[start + dim] * weight;
            ders[dim].Der1 = expApprox;
            sumExpApprox += expApprox;
        }
        for (int dim = offset; dim < offset + count; ++dim) {
            const float weight = weights.empty() ? 1.0f : weights[start + dim];
            if (weight > 0) {
                const double p = ders[dim].Der1 / sumExpApprox;
                ders[dim].Der2 = Beta * sumWeightedTargets * (Beta * p * (p - 1.0) - LambdaReg);
                ders[dim].Der1 = Beta * (-sumWeightedTargets * p + weight * targets[start + dim]);
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

void CheckDerivativeOrderForObjectImportance(ui32 derivativeOrder, ELeavesEstimation estimationMethod) {
    CB_ENSURE(derivativeOrder >= 2, "Current error function doesn't support object importance calculation");
    if (estimationMethod == ELeavesEstimation::Newton) {
        CB_ENSURE(
            derivativeOrder >= 3,
            "Current error function doesn't support object importance calculation with Newton leaves"
            " estimation method");
    }
}

// TLambdaMartError definitions
TLambdaMartError::TLambdaMartError(
    ELossFunction targetMetric,
    const TMap<TString, TString>& metricParams,
    double sigma, bool norm)
    : IDerCalcer(false, 1, EErrorType::QuerywiseError)
    , TargetMetric(targetMetric)
    , TopSize(NCatboostOptions::GetParamOrDefault(metricParams, "top", -1))
    , NumeratorType(NCatboostOptions::GetParamOrDefault(metricParams, "type", ENdcgMetricType::Base))
    , DenominatorType(NCatboostOptions::GetParamOrDefault(metricParams, "denominator", ENdcgDenominatorType::LogPosition))
    , Sigma(sigma)
    , Norm(norm)
{
    CB_ENSURE(EqualToOneOf(TargetMetric, ELossFunction::DCG, ELossFunction::NDCG, ELossFunction::MRR, ELossFunction::ERR, ELossFunction::MAP),
        "Only DCG, NDCG, MRR, ERR and MAP target metric supported for LambdaMART now");
    CB_ENSURE(Sigma > 0, "Sigma should be positive");
}

void TLambdaMartError::CalcDersForQueries(
    int queryStartIndex,
    int queryEndIndex,
    const TVector<double>& approxes,
    const TVector<float>& target,
    const TVector<float>& /*weights*/,
    const TVector<TQueryInfo>& queriesInfo,
    TArrayRef<TDers> ders,
    ui64 /*randomSeed*/,
    NPar::ILocalExecutor* localExecutor
) const {
    auto start = queriesInfo[queryStartIndex].Begin;
    NPar::ParallelFor(*localExecutor, queryStartIndex, queryEndIndex, [&](int queryIndex) {
        auto begin = queriesInfo[queryIndex].Begin;
        auto end = queriesInfo[queryIndex].End;
        auto count = end - begin;
        TArrayRef<TDers> queryDers(ders.data() + begin - start, count);
        TConstArrayRef<double> queryApproxes(approxes.data() + begin, count);
        TConstArrayRef<float> queryTargets(target.data() + begin, count);
        CalcDersForSingleQuery(queryApproxes, queryTargets, queryDers);
    });
}

void TLambdaMartError::CalcDersNDCG(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> targets,
    TConstArrayRef<size_t> order,
    TArrayRef<TDers> ders,
    double& sumDer1
) const {
    const size_t count = approxes.size();
    const size_t queryTopSize = GetQueryTopSize(count);
    const double idealScore = CalcIdealMetric(targets, queryTopSize);
    const bool isApproxesSame = (approxes[order[0]] == approxes[order[count - 1]]);

    for (size_t firstId = 0; firstId < count; ++firstId) {
        size_t boundForSecondId = firstId < queryTopSize ? count : queryTopSize;
        for (size_t secondId = 0; secondId < boundForSecondId; ++secondId) {

            const double firstTarget = targets[order[firstId]];
            const double secondTarget = targets[order[secondId]];

            if (firstTarget <= secondTarget) {
                continue;
            }

            const double approxDiff = approxes[order[firstId]] - approxes[order[secondId]];

            const double dcgNum = CalcNumerator(firstTarget) - CalcNumerator(secondTarget);
            const double dcgDen = std::abs(1.0 / CalcDenominator(firstId) - 1.0 / CalcDenominator(secondId));

            double delta = dcgNum * dcgDen / idealScore;

            if (Norm && !isApproxesSame) {
                delta /= 0.01 + std::abs(approxDiff);
            }

            double antigrad = 1.0 / (1.0 + std::exp(Sigma * approxDiff));
            double hessian = antigrad * (1 - antigrad);
            antigrad *=  - Sigma * delta;
            hessian *= Sigma * Sigma * delta;

            ders[order[firstId]].Der1 += antigrad;
            ders[order[firstId]].Der2 += hessian;
            ders[order[secondId]].Der1 -= antigrad;
            ders[order[secondId]].Der2 += hessian;

            sumDer1 -= 2 * antigrad;
        }
    }
}

void TLambdaMartError::CalcDersMRR(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> targets,
    TConstArrayRef<size_t> order,
    TArrayRef<TDers> ders,
    double& sumDer1
) const {
    const size_t count = approxes.size();
    const bool isApproxesSame = (approxes[order[0]] == approxes[order[count - 1]]);
    size_t numRelevants = 0;
    size_t firstRelevantId = 0;
    size_t secondRelevantId = 0;
    for (size_t id = 0; id < count; ++id) {
        if (targets[order[id]] > 0) {
            ++numRelevants;
            if (numRelevants == 1) {
                firstRelevantId = id;
            } else if (numRelevants == 2) {
                secondRelevantId = id;
                break;
            }
        }
    }
    if (!numRelevants) {
        return;
    }

    for (size_t firstId = 0; firstId < count; ++firstId) {
        const bool isFirstRelevant = targets[order[firstId]] > 0;
        if (!isFirstRelevant) {
            continue;
        }

        size_t boundForSecondId = firstId == firstRelevantId ? count : firstRelevantId;
        for (size_t secondId = 0; secondId < boundForSecondId; ++secondId) {
            const bool isSecondRelevant = targets[order[secondId]] > 0;

            if (isSecondRelevant) {
                continue;
            }

            const double approxDiff = approxes[order[firstId]] - approxes[order[secondId]];

            double delta = secondId < firstId
                ? 1.0 / (secondId + 1) - 1.0 / (firstId + 1)
                : (secondRelevantId == 0 || secondId < secondRelevantId
                    ? 1.0 / (firstId + 1) - 1.0 / (secondId + 1)
                    : 1.0 / (firstId + 1) - 1.0 / (secondRelevantId + 1)
                );

            if (Norm && !isApproxesSame) {
                delta /= 0.01 + std::abs(approxDiff);
            }

            double antigrad = 1.0 / (1.0 + std::exp(Sigma * approxDiff));
            double hessian = antigrad * (1 - antigrad);
            antigrad *=  - Sigma * delta;
            hessian *= Sigma * Sigma * delta;

            ders[order[firstId]].Der1 += antigrad;
            ders[order[firstId]].Der2 += hessian;
            ders[order[secondId]].Der1 -= antigrad;
            ders[order[secondId]].Der2 += hessian;

            sumDer1 -= 2 * antigrad;
        }
    }
}

void TLambdaMartError::CalcDersERR(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> targets,
    TConstArrayRef<size_t> order,
    TArrayRef<TDers> ders,
    double& sumDer1
) const {
    const size_t count = approxes.size();
    const bool isApproxesSame = (approxes[order[0]] == approxes[order[count - 1]]);

    double pFirstLook = 1.0;
    for (size_t firstId = 0; firstId < count; ++firstId) {
        const int firstCandidate = order[firstId];

        double pMiddleLook = 1.0;
        double middleRR = 0.0;
        for (size_t secondId = firstId + 1; secondId < count; ++secondId) {
            const int secondCandidate = order[secondId];

            if (targets[firstCandidate] == targets[secondCandidate])
                continue;

            const size_t highId = targets[firstCandidate] > targets[secondCandidate] ? firstId : secondId;
            const size_t lowId = targets[firstCandidate] > targets[secondCandidate] ? secondId : firstId;

            const double approxDiff = approxes[order[highId]] - approxes[order[lowId]];

            const double firstDelta = (targets[firstCandidate] - targets[secondCandidate]) / (firstId + 1);
            const double middleDelta = middleRR * (targets[secondCandidate] - targets[firstCandidate]);
            const double secondDelta = pMiddleLook * (targets[secondCandidate] - targets[firstCandidate]) / (secondId + 1);

            double delta = pFirstLook * std::abs(firstDelta + middleDelta + secondDelta);

            middleRR += pMiddleLook * targets[secondCandidate] / (secondId + 1);
            pMiddleLook *= (1 - targets[secondCandidate]);

            if (Norm && !isApproxesSame) {
                delta /= 0.01 + std::abs(approxDiff);
            }

            double antigrad = 1.0 / (1.0 + std::exp(Sigma * approxDiff));
            double hessian = antigrad * (1 - antigrad);
            antigrad *=  - Sigma * delta;
            hessian *= Sigma * Sigma * delta;

            ders[order[highId]].Der1 += antigrad;
            ders[order[highId]].Der2 += hessian;
            ders[order[lowId]].Der1 -= antigrad;
            ders[order[lowId]].Der2 += hessian;

            sumDer1 -= 2 * antigrad;
        }

        pFirstLook *= 1 - targets[firstCandidate];
    }
}

void TLambdaMartError::CalcDersMAP(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> targets,
    TConstArrayRef<size_t> order,
    TArrayRef<TDers> ders,
    double& sumDer1
) const {
    const size_t count = approxes.size();
    const bool isApproxesSame = (approxes[order[0]] == approxes[order[count - 1]]);

    for (size_t firstId = 0; firstId < count; ++firstId) {
        const int firstCandidate = order[firstId];
        const bool isFirstRelevant = targets[firstCandidate] > 0;

        double sumRR = 1.0 / (firstId + 1);

        for (size_t secondId = firstId + 1; secondId < count; ++secondId) {
            const int secondCandidate = order[secondId];
            const bool isSecondRelevant = targets[secondCandidate] > 0;

            if (isFirstRelevant == isSecondRelevant)
                continue;

            const size_t highId = targets[firstCandidate] > targets[secondCandidate] ? firstId : secondId;
            const size_t lowId = targets[firstCandidate] > targets[secondCandidate] ? secondId : firstId;

            const double approxDiff = approxes[order[highId]] - approxes[order[lowId]];

            double delta = sumRR;

            if (Norm && !isApproxesSame) {
                delta /= 0.01 + std::abs(approxDiff);
            }

            double antigrad = 1.0 / (1.0 + std::exp(Sigma * approxDiff));
            double hessian = antigrad * (1 - antigrad);
            antigrad *=  - Sigma * delta;
            hessian *= Sigma * Sigma * delta;

            ders[order[highId]].Der1 += antigrad;
            ders[order[highId]].Der2 += hessian;
            ders[order[lowId]].Der1 -= antigrad;
            ders[order[lowId]].Der2 += hessian;

            sumDer1 -= 2 * antigrad;

            if (isSecondRelevant) {
                sumRR += 1.0 / (secondId + 1);
            }
        }

    }
}

void TLambdaMartError::CalcDersForSingleQuery(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> targets,
    TArrayRef<TDers> ders
) const {
    const size_t count = approxes.size();
    Y_ASSERT(targets.size() == count);
    Y_ASSERT(ders.size() == count);

    Fill(ders.begin(), ders.end(), TDers{0.0, 0.0, 0.0});

    if (count <= 1) {
        return;
    }

    TVector<size_t> order(count);
    Iota(order.begin(), order.end(), 0);
    StableSort(order.begin(), order.end(), [&](int a, int b) {
        return approxes[a] > approxes[b];
    });

    double sumDer1 = 0.0;

    if (TargetMetric == ELossFunction::DCG || TargetMetric == ELossFunction::NDCG) {
        CalcDersNDCG(
            approxes,
            targets,
            order,
            ders,
            sumDer1
        );
    } else if (TargetMetric == ELossFunction::MRR) {
        CalcDersMRR(
            approxes,
            targets,
            order,
            ders,
            sumDer1
        );
    } else if (TargetMetric == ELossFunction::ERR) {
        CalcDersERR(
            approxes,
            targets,
            order,
            ders,
            sumDer1
        );
    } else if (TargetMetric == ELossFunction::MAP) {
        CalcDersMAP(
            approxes,
            targets,
            order,
            ders,
            sumDer1
        );
    }

    if (Norm && sumDer1 > 0) {
        double norma = std::log2(1 + sumDer1) / sumDer1;
        for (auto& der : ders) {
            der.Der1 *= norma;
            der.Der2 *= norma;
        }
    }
}

double TLambdaMartError::CalcIdealMetric(TConstArrayRef<float> target, size_t queryTopSize) const {
    double score = 0;
    TVector<float> sortedTargets(target.begin(), target.end());
    StableSort(sortedTargets, [](float a, float b) {
        return a > b;
    });
    for (size_t id = 0; id < queryTopSize; ++id) {
        score += CalcNumerator(sortedTargets[id])  / CalcDenominator(id);
    }
    return score;
}

static inline ENdcgDenominatorType GetDefaultDenominatorType(ELossFunction metric) {
    return metric == ELossFunction::FilteredDCG
        ? ENdcgDenominatorType::Position
        : ENdcgDenominatorType::LogPosition;
}

// TStochasticRankError definitions
TStochasticRankError::TStochasticRankError(
    ELossFunction targetMetric,
    const TMap<TString, TString>& metricParams,
    double sigma,
    size_t numEstimations,
    double mu,
    double nu,
    double lambda)
    : IDerCalcer(false, 1, EErrorType::QuerywiseError)
    , TargetMetric(targetMetric)
    , TopSize(NCatboostOptions::GetParamOrDefault(metricParams, "top", -1))
    , NumeratorType(NCatboostOptions::GetParamOrDefault(metricParams, "type", ENdcgMetricType::Base))
    , DenominatorType(NCatboostOptions::GetParamOrDefault(metricParams, "denominator", GetDefaultDenominatorType(targetMetric)))
    , Decay(NCatboostOptions::GetParamOrDefault(metricParams, "decay", 0.85))
    , Sigma(sigma)
    , NumEstimations(numEstimations)
    , Mu(mu)
    , Nu(nu)
    , Lambda(lambda)
{
    CB_ENSURE(EqualToOneOf(TargetMetric, ELossFunction::DCG, ELossFunction::NDCG,
                                         ELossFunction::PFound, ELossFunction::FilteredDCG, ELossFunction::ERR, ELossFunction::MRR),
        "Only DCG, NDCG, PFound, FilteredDCG, ERR and MRR target metric supported for StochasticRank now");
    CB_ENSURE(0.0 <= Decay && Decay <= 1.0, "Decay should be in [0, 1]");
    CB_ENSURE(NumEstimations > 0, "Number of estimations should be positive");
    CB_ENSURE(Sigma > 0, "Sigma should be positive");
    CB_ENSURE(Mu >= 0, "Mu should be non-negative");
    CB_ENSURE(Nu > 0, "Nu should be positive");
}

void TStochasticRankError::CalcDersForQueries(
    int queryStartIndex,
    int queryEndIndex,
    const TVector<double>& approxes,
    const TVector<float>& target,
    const TVector<float>& /*weights*/,
    const TVector<TQueryInfo>& queriesInfo,
    TArrayRef<TDers> ders,
    ui64 randomSeed,
    NPar::ILocalExecutor* localExecutor
) const {
    auto start = queriesInfo[queryStartIndex].Begin;
    NPar::ParallelFor(*localExecutor, queryStartIndex, queryEndIndex, [&](int queryIndex) {
        auto begin = queriesInfo[queryIndex].Begin;
        auto end = queriesInfo[queryIndex].End;
        auto count = end - begin;
        if (TargetMetric == ELossFunction::FilteredDCG) {
            // we add virtual document with zero approx and target to reuse for DCG
            TVector<TDers> queryDers(count + 1);
            TVector<double> queryApproxes(count + 1);
            TVector<float> queryTargets(count + 1);
            CopyN(approxes.data() + begin, count, queryApproxes.data());
            CopyN(target.data() + begin, count, queryTargets.data());
            CalcDersForSingleQuery(queryApproxes, queryTargets, randomSeed + queryIndex, queryDers);
            CopyN(queryDers.data(), count, ders.data() + begin - start);
        } else {
            TArrayRef<TDers> queryDers(ders.data() + begin - start, count);
            TConstArrayRef<double> queryApproxes(approxes.data() + begin, count);
            TConstArrayRef<float> queryTargets(target.data() + begin, count);
            CalcDersForSingleQuery(queryApproxes, queryTargets, randomSeed + queryIndex, queryDers);
        }
    });
}

void TStochasticRankError::CalcDersForSingleQuery(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> targets,
    ui64 randomSeed,
    TArrayRef<TDers> ders
) const {
    size_t count = approxes.size();
    Y_ASSERT(targets.size() == count);
    Y_ASSERT(ders.size() == count);

    Fill(ders.begin(), ders.end(), TDers{0.0, 0.0, 0.0});

    if (count <= 1) {
        return;
    }

    // Stage 1 - shift approxes to break ties
    TVector<double> shiftedApproxes(count);
    for (size_t docId = 0; docId < count; ++docId) {
        shiftedApproxes[docId] = approxes[docId] - Sigma * Mu * targets[docId];
    }
    if (TargetMetric != ELossFunction::FilteredDCG) {
        double avrgShiftedApprox = Accumulate(shiftedApproxes, 0.0) / count;
        for (size_t docId = 0; docId < count; ++docId) {
            shiftedApproxes[docId] -= avrgShiftedApprox;
        }
    }

    // Stage 2 - estimate gradients via noise and Monte Carlo method
    TVector<double> posWeights;
    TVector<double> noise(count);
    TVector<double> scores(count);
    TVector<size_t> order(count);
    TFastRng64 rng(randomSeed);
    for (size_t sample = 0; sample < NumEstimations; ++sample) {
        for(size_t docId = 0; docId < count; ++docId) {
            noise[docId] = StdNormalDistribution<double>(rng);
            scores[docId] = shiftedApproxes[docId] + Sigma * noise[docId];
        }
        if (TargetMetric == ELossFunction::FilteredDCG) {
            noise.back() = 0;
            scores.back() = shiftedApproxes.back();
        }
        const double noiseSum = Accumulate(noise, 0.0);
        Iota(order.begin(), order.end(), 0);
        StableSort(order.begin(), order.end(), [&](int a, int b) {
            return scores[a] > scores[b];
        });
        if (EqualToOneOf(TargetMetric, ELossFunction::DCG, ELossFunction::NDCG, ELossFunction::FilteredDCG) && sample == 0) {
            posWeights = ComputeDCGPosWeights(targets);
        } else if (TargetMetric == ELossFunction::PFound) {
            posWeights = ComputePFoundPosWeights(targets, order);
        } else if (TargetMetric == ELossFunction::ERR) {
            posWeights = ComputeERRPosWeights(targets, order);
        }
        CalcMonteCarloEstimateForSingleQueryPermutation(
            targets,
            shiftedApproxes,
            scores,
            order,
            posWeights,
            noiseSum,
            ders
        );
    }

    // Stage 3 - SFA, make gradients ortogonal with approxes
    if (TargetMetric != ELossFunction::FilteredDCG) {
        double avrgDer = 0.0;
        for (const auto& der : ders) {
            avrgDer += der.Der1;
        }
        avrgDer /= count;
        for (auto& der : ders) {
            der.Der1 -= avrgDer;
        }
    }
    if (count > 2) {
        double avrgApprox = Accumulate(approxes, 0.0) / count;
        TVector<double> zeroMeanApproxes(count);
        for (size_t docId = 0; docId < count; ++docId) {
            zeroMeanApproxes[docId] = approxes[docId] - avrgApprox;
        }
        double approxesNormSqr = 0.0;
        for (size_t docId = 0; docId < count; ++docId) {
            approxesNormSqr += Sqr(zeroMeanApproxes[docId]);
        }
        approxesNormSqr = Sqr(std::sqrt(approxesNormSqr) + Nu);
        double dot = 0.0;
        for (size_t docId = 0; docId < count; ++docId) {
            dot += ders[docId].Der1 * zeroMeanApproxes[docId];
        }
        const double k = Lambda * dot / approxesNormSqr;
        for (size_t docId = 0; docId < count; ++docId) {
            ders[docId].Der1 -= k * zeroMeanApproxes[docId];
        }
    }
}

void TStochasticRankError::CalcMonteCarloEstimateForSingleQueryPermutation(
    TConstArrayRef<float> targets,
    const TVector<double>& approxes,
    const TVector<double>& scores,
    const TVector<size_t>& order,
    const TVector<double>& posWeights,
    const double noiseSum,
    TArrayRef<TDers> ders
) const {
    const size_t count = targets.size();
    const size_t queryTopSize = GetQueryTopSize(count);

    TVector<double> cumSum(count + 1);
    TVector<double> cumSumUp(count + 1);
    TVector<double> cumSumLow(count + 1);
    int firstRelevPos = -1, secondRelevPos = -1; // avoid passing uninitialized values
    if (EqualToOneOf(TargetMetric, ELossFunction::DCG, ELossFunction::NDCG, ELossFunction::FilteredDCG)) {
        CalcDCGCumulativeStatistics(targets, order, posWeights, scores, cumSum, cumSumUp, cumSumLow);
    } else if (TargetMetric == ELossFunction::PFound) {
        CalcPFoundCumulativeStatistics(targets, order, posWeights, cumSum);
    } else if (TargetMetric == ELossFunction::ERR) {
        CalcERRCumulativeStatistics(targets, order, posWeights, cumSum, cumSumUp, cumSumLow);
    } else if (TargetMetric == ELossFunction::MRR) {
        CalcMRRStatistics(targets, order, &firstRelevPos, &secondRelevPos);
    } else {
        CB_ENSURE(false, "StochasticRank is unimplemented for " << TargetMetric);
    }

    auto calcDerSumOverAllPositions = [&](size_t pos, double mean, double sigma, size_t maxNewPos) {
        double derSum = 0.0;
        for (size_t newPos = 0; newPos < maxNewPos + 1; ++newPos) {
            if (newPos == pos) {
                continue;
            }
            const double metricDiff = CalcMetricDiff(pos, newPos, queryTopSize, targets, order,
                                                     posWeights, scores, cumSum, cumSumUp, cumSumLow,
                                                     firstRelevPos, secondRelevPos);
            double densityDiff = 0.0;
            if (newPos == 0) {
                densityDiff = NormalDensity(scores[order[0]], mean, sigma);
            } else if (newPos + 1 == Min(count, queryTopSize + 1)) {
                densityDiff = newPos < pos
                    ? -NormalDensity(scores[order[queryTopSize - 1]], mean, sigma)
                    : -NormalDensity(scores[order[Min(queryTopSize, count - 1)]], mean, sigma);
            } else {
                densityDiff = newPos < pos
                    ? NormalDensityDiff(scores[order[newPos]], scores[order[newPos - 1]], mean, sigma)
                    : NormalDensityDiff(scores[order[newPos + 1]], scores[order[newPos]], mean, sigma);
            }
            derSum += metricDiff * densityDiff;
        }
        return derSum;
    };

    for (size_t pos = 0; pos < count; ++pos) {
        const size_t docId = order[pos];
        const double score = scores[docId];
        const double approx = approxes[docId];
        const double mean = approx + (
            TargetMetric == ELossFunction::FilteredDCG ? 0 : (noiseSum - (score - approx)) / (count - 1)
        );
        const double sigma = Sigma * (
            TargetMetric == ELossFunction::FilteredDCG ? 1 : std::sqrtl(count / (count - 1.0))
        );
        double derSum = 0.0;
        if (EqualToOneOf(TargetMetric, ELossFunction::DCG, ELossFunction::NDCG, ELossFunction::PFound, ELossFunction::FilteredDCG, ELossFunction::ERR)) {
            derSum += calcDerSumOverAllPositions(pos, mean, sigma, Min(count - 1, queryTopSize));
        } else if (TargetMetric == ELossFunction::MRR) {
            if (static_cast<int>(pos) == firstRelevPos) {
                if (secondRelevPos == -1) {
                    derSum += calcDerSumOverAllPositions(pos, mean, sigma, count - 1);
                } else {
                    derSum += calcDerSumOverAllPositions(pos, mean, sigma, secondRelevPos - 1);
                    // newPos >= secondRelevPos
                    {
                        const double metricDiff = 1.0 / secondRelevPos - 1.0 / (firstRelevPos + 1);
                        const double densityDiff = -NormalDensity(scores[order[secondRelevPos]], mean, sigma);
                        derSum += metricDiff * densityDiff;
                    }
                }
            }
            else if (static_cast<int>(pos) == secondRelevPos) {
                derSum += calcDerSumOverAllPositions(pos, mean, sigma, firstRelevPos);
            }
            else if (static_cast<int>(pos) < firstRelevPos) {
                // newPos >= firstRelevPos
                {
                    const double metricDiff = 1.0 / firstRelevPos - 1.0 / (firstRelevPos + 1);
                    const double densityDiff = -NormalDensity(scores[order[firstRelevPos]], mean, sigma);
                    derSum += metricDiff * densityDiff;
                }
            }
            else if (static_cast<int>(pos) < secondRelevPos) {
                // newPos <= firstRelevPos
                {
                    const double metricDiff = 1.0 / (firstRelevPos + 2) - 1.0 / (firstRelevPos + 1);
                    const double densityDiff = NormalDensity(scores[order[firstRelevPos]], mean, sigma);
                    derSum += metricDiff * densityDiff;
                }
            }
            else if (firstRelevPos != -1) {
                Y_ASSERT(static_cast<int>(pos) > secondRelevPos);
                // newPos <= firstRelevPos
                if (targets[order[pos]] == 0) {
                    const double densityDiff = NormalDensity(scores[order[firstRelevPos]], mean, sigma);
                    const double metricDiff = 1.0 / (firstRelevPos + 2) - 1.0 / (firstRelevPos + 1);
                    derSum += metricDiff * densityDiff;
                } else {
                    derSum += calcDerSumOverAllPositions(pos, mean, sigma, firstRelevPos);
                }
            }
        }
        ders[docId].Der1 += derSum / NumEstimations;
    }
}
double TStochasticRankError::CalcDCGMetricDiff(
    size_t oldPos,
    size_t newPos,
    const TConstArrayRef<float> targets,
    const TVector<size_t>& order,
    const TVector<double>& posWeights,
    const TVector<double>& scores,
    const TVector<double>& cumSum,
    const TVector<double>& cumSumUp,
    const TVector<double>& cumSumLow
) const {
    double oldWeight = posWeights[oldPos];
    double newWeight = posWeights[newPos];
    if (TargetMetric == ELossFunction::FilteredDCG) {
        const double oldScore = scores[order[oldPos]];
        oldWeight *= (oldScore >= 0);
        const double newScore = newPos < oldPos
            ? scores[order[newPos]]
            : (newPos + 1 < scores.size() ? scores[order[newPos + 1]] : -1);
        newWeight *= (newScore >= 0);
    }
    const double docGain = CalcNumerator(targets[order[oldPos]]);
    const double docDiff = docGain * (newWeight - oldWeight);
    double midDiff = 0.0;
    if (newPos < oldPos) {
        const double oldMidValue = cumSum[oldPos] - cumSum[newPos];
        const double newMidValue = cumSumLow[oldPos] - cumSumLow[newPos];
        midDiff = newMidValue - oldMidValue;
    } else {
        const double oldMidValue = cumSum[newPos + 1] - cumSum[oldPos + 1];
        const double newMidValue = cumSumUp[newPos + 1] - cumSumUp[oldPos + 1];
        midDiff = newMidValue - oldMidValue;
    }
    return docDiff + midDiff;
}

double TStochasticRankError::CalcPFoundMetricDiff(
    size_t oldPos,
    size_t newPos,
    size_t queryTopSize,
    const TConstArrayRef<float> targets,
    const TVector<size_t>& order,
    const TVector<double>& posWeights,
    const TVector<double>& cumSum
) const {
    const double docGain = targets[order[oldPos]];
    double docDiff = 0.0;
    double midDiff = 0.0;
    if (newPos < oldPos) {
        const double oldWeight = posWeights[oldPos];
        const double newWeight = posWeights[newPos];
        docDiff = docGain * (newWeight - oldWeight);
        const double oldMidValue = cumSum[oldPos] - cumSum[newPos];
        double newMidValue = (cumSum[oldPos] - cumSum[newPos]) * Decay * (1 - docGain);
        if (oldPos >= queryTopSize) {
            const double lastValue = posWeights[queryTopSize - 1] * targets[order[queryTopSize - 1]];
            newMidValue -= lastValue * Decay * (1 - docGain);
        }
        midDiff = newMidValue - oldMidValue;
    } else {
        const double oldWeight = posWeights[oldPos];
        const double oldMidValue = cumSum[newPos + 1] - cumSum[oldPos + 1];
        double newWeight = 0.0;
        double newMidValue = 0.0;
        if (Decay == 0.0) { // only first item has non-zero contribution
            newWeight = 0.0;
            newMidValue = oldPos == 0 ? targets[order[1]] : oldMidValue;
        } else if (docGain == 1.0) { // we can't use cumsum because of zero multipliers
            double plook = posWeights[oldPos];
            for (size_t pos = oldPos + 1; pos <= Min(newPos, queryTopSize - 1); ++pos) {
                newMidValue += targets[order[pos]] * plook;
                plook *= (1 - targets[order[pos]]) * Decay;
            }
            newWeight = newPos < queryTopSize ? plook : 0.0;
        } else {
            newWeight = posWeights[newPos] * (1 - targets[order[newPos]]) / (1 - docGain);
            newMidValue = (cumSum[newPos + 1] - cumSum[oldPos + 1]) / Decay / (1 - docGain);
            if (newPos >= queryTopSize) {
                const double lastValue = posWeights[queryTopSize - 1] * targets[order[queryTopSize]];
                newMidValue += lastValue / (1 - docGain) * (1 - targets[order[queryTopSize - 1]]);
            }
        }
        docDiff = docGain * (newWeight - oldWeight);
        midDiff = newMidValue - oldMidValue;
    }
    return docDiff + midDiff;
}

double TStochasticRankError::CalcERRMetricDiff(
    size_t oldPos,
    size_t newPos,
    size_t queryTopSize,
    const TConstArrayRef<float> targets,
    const TVector<size_t>& order,
    const TVector<double>& posWeights,
    const TVector<double>& cumSum,
    const TVector<double>& cumSumUp,
    const TVector<double>& cumSumLow
) const {
    const double docGain = targets[order[oldPos]];
    double docDiff = 0.0;
    double midDiff = 0.0;
    if (newPos < oldPos) {
        const double oldWeight = posWeights[oldPos];
        const double newWeight = posWeights[newPos];
        docDiff = docGain * (newWeight - oldWeight);
        const double oldMidValue = cumSum[oldPos] - cumSum[newPos];
        double newMidValue = (cumSumLow[oldPos] - cumSumLow[newPos]) * (1 - docGain);
        if (oldPos >= queryTopSize) {
            const double lastValue = posWeights[queryTopSize - 1] * targets[order[queryTopSize - 1]];
            newMidValue -= lastValue * (1 - docGain);
        }
        midDiff = newMidValue - oldMidValue;
    } else {
        const double oldWeight = posWeights[oldPos];
        const double oldMidValue = cumSum[newPos + 1] - cumSum[oldPos + 1];
        double newWeight = 0.0;
        double newMidValue = 0.0;
        if (docGain == 1.0) { // we can't use cumsum because of zero multipliers
            double plook = posWeights[oldPos] * (oldPos + 1);
            for (size_t pos = oldPos + 1; pos <= newPos && plook > 0; ++pos) {
                newMidValue += targets[order[pos]] * plook / pos;
                plook *= (1 - targets[order[pos]]);
            }
            newWeight = newPos < queryTopSize ? plook / (newPos + 1) : 0.0;
        } else {
            newWeight = posWeights[newPos] * (1 - targets[order[newPos]]) / (1 - docGain) * (newPos + 2.0) / (newPos + 1.0);
            newMidValue = (cumSumUp[newPos + 1] - cumSumUp[oldPos + 1]) / (1 - docGain);
            if (newPos >= queryTopSize) {
                const double lastValue = posWeights[queryTopSize - 1] * targets[order[queryTopSize]];
                newMidValue += lastValue / (1 - docGain) * (1 - targets[order[queryTopSize - 1]]);
            }
        }
        docDiff = docGain * (newWeight - oldWeight);
        midDiff = newMidValue - oldMidValue;
    }
    return docDiff + midDiff;
}

double TStochasticRankError::CalcMRRMetricDiff(
    size_t oldPos,
    size_t newPos,
    const TConstArrayRef<float> targets,
    const TVector<size_t>& order,
    int firstRelevPos,
    int secondRelevPos
) const {
    if (firstRelevPos == -1) {
        return 0;
    }

    if (oldPos < static_cast<size_t>(firstRelevPos)) {
        if (newPos >= static_cast<size_t>(firstRelevPos)) {
            return 1.0 / firstRelevPos  - 1.0 / (firstRelevPos + 1);
        } else {
            return 0.0;
        }
    }
    if (oldPos == static_cast<size_t>(firstRelevPos)) {
        if (secondRelevPos == -1 || newPos < static_cast<size_t>(secondRelevPos)) {
            return 1.0 / (newPos + 1) - 1.0 / (oldPos + 1);
        } else {
            return 1.0 / secondRelevPos - 1.0 / (oldPos + 1);
        }
    }
    if (newPos <= static_cast<size_t>(firstRelevPos)) {
        if (targets[order[oldPos]] > 0) {
            return 1.0 / (newPos + 1) - 1.0 / (firstRelevPos + 1);
        } else {
            return 1.0 / (firstRelevPos + 2)  - 1.0 / (firstRelevPos + 1);
        }
    }

    return 0;
}

double TStochasticRankError::CalcMetricDiff(
    size_t oldPos,
    size_t newPos,
    size_t queryTopSize,
    const TConstArrayRef<float> targets,
    const TVector<size_t>& order,
    const TVector<double>& posWeights,
    const TVector<double>& scores,
    const TVector<double>& cumSum,
    const TVector<double>& cumSumUp,
    const TVector<double>& cumSumLow,
    int firstRelevPos,
    int secondRelevPos
) const {
    if (newPos == oldPos || Min(oldPos, newPos) >= queryTopSize) {
        return 0.0;
    }

    if (EqualToOneOf(TargetMetric, ELossFunction::DCG, ELossFunction::NDCG, ELossFunction::FilteredDCG)) {
        return CalcDCGMetricDiff(oldPos, newPos, targets, order, posWeights, scores, cumSum, cumSumUp, cumSumLow);
    } else if (TargetMetric == ELossFunction::PFound) {
        return CalcPFoundMetricDiff(oldPos, newPos, queryTopSize, targets, order, posWeights, cumSum);
    } else if (TargetMetric == ELossFunction::ERR) {
        return CalcERRMetricDiff(oldPos, newPos, queryTopSize, targets, order, posWeights, cumSum, cumSumUp, cumSumLow);
    } else if (TargetMetric == ELossFunction::MRR) {
        return CalcMRRMetricDiff(oldPos, newPos, targets, order, firstRelevPos, secondRelevPos);
    }
    CB_ENSURE(false, "Unexpected target metric type");
}

void TStochasticRankError::CalcDCGCumulativeStatistics(
    TConstArrayRef<float> targets,
    const TVector<size_t>& order,
    const TVector<double>& posWeights,
    const TVector<double>& scores,
    TArrayRef<double> cumSum,
    TArrayRef<double> cumSumUp,
    TArrayRef<double> cumSumLow
) const {
    const size_t count = targets.size();
    cumSum[0] = cumSumUp[0] = cumSumLow[0] = cumSumUp[1] = 0;
    for (size_t pos = 0; pos < count; ++pos) {
        const size_t docId = order[pos];
        const double gain = (TargetMetric == ELossFunction::FilteredDCG && scores[docId] < 0) ? 0 : CalcNumerator(targets[docId]);
        cumSum[pos + 1] = cumSum[pos] + gain * posWeights[pos];
        if (pos + 1 < count) {
            cumSumLow[pos + 1] = cumSumLow[pos] + gain * posWeights[pos + 1];
        }
        if (pos > 0) {
            cumSumUp[pos + 1] = cumSumUp[pos] + gain * posWeights[pos - 1];
        }
    }
    cumSumLow[count] = cumSumLow[count - 1];
}

void TStochasticRankError::CalcPFoundCumulativeStatistics(
    TConstArrayRef<float> targets,
    const TVector<size_t>& order,
    const TVector<double>& posWeights,
    TArrayRef<double> cumSum
) const {
    const size_t count = targets.size();
    cumSum[0] = 0;
    for (size_t pos = 0; pos < count; ++pos) {
        const size_t docId = order[pos];
        const double gain = targets[docId];
        cumSum[pos + 1] = cumSum[pos] + gain * posWeights[pos];
    }
}

void TStochasticRankError::CalcERRCumulativeStatistics(
    TConstArrayRef<float> targets,
    const TVector<size_t>& order,
    const TVector<double>& posWeights,
    TArrayRef<double> cumSum,
    TArrayRef<double> cumSumUp,
    TArrayRef<double> cumSumLow
) const {
    const size_t count = targets.size();
    cumSum[0] = cumSumUp[0] = cumSumLow[0] = cumSumUp[1] = 0;
    for (size_t pos = 0; pos < count; ++pos) {
        const size_t docId = order[pos];
        const double gain = targets[docId];
        cumSum[pos + 1] = cumSum[pos] + gain * posWeights[pos];
        if (pos + 1 < count) {
            cumSumLow[pos + 1] = cumSumLow[pos] + gain * posWeights[pos] * (pos + 1.0) / (pos + 2.0);
        }
        if (pos > 0) {
            cumSumUp[pos + 1] = cumSumUp[pos] + gain * posWeights[pos] * (pos + 1.0) / (pos + 0.0);
        }
    }
}



void TStochasticRankError::CalcMRRStatistics(
    TConstArrayRef<float> targets,
    const TVector<size_t>& order,
    int* firstRelevPos,
    int* secondRelevPos
) const {
    *firstRelevPos = -1;
    *secondRelevPos = -1;
    for (size_t pos = 0; pos < order.size(); ++pos) {
        size_t docId = order[pos];
        if (targets[docId] > 0) {
            if (*firstRelevPos == -1) {
                *firstRelevPos = pos;
            } else if (*secondRelevPos == -1) {
                *secondRelevPos = pos;
                break;
            }
        }
    }
}

TVector<double> TStochasticRankError::ComputeDCGPosWeights(
    TConstArrayRef<float> targets
) const {
    size_t count = targets.size();
    TVector<double> posWeights(count);
    size_t queryTopSize = GetQueryTopSize(count);
    Y_ASSERT(EqualToOneOf(TargetMetric, ELossFunction::DCG, ELossFunction::NDCG, ELossFunction::FilteredDCG));
    for (size_t pos = 0; pos < queryTopSize; ++pos) {
        posWeights[pos] = 1.0 / CalcDenominator(pos);
    }

    if (TargetMetric == ELossFunction::NDCG) {
        TVector<float> sortedTargets(targets.begin(), targets.end());
        StableSort(sortedTargets, [](float a, float b) {
            return a > b;
        });
        const double idealDCG = CalcDCG(sortedTargets, posWeights);
        if (idealDCG > std::numeric_limits<double>::epsilon()) {
            for (size_t pos = 0; pos < queryTopSize; ++pos) {
                posWeights[pos] /= idealDCG;
            }
        }
    }
    return posWeights;
}

TVector<double> TStochasticRankError::ComputePFoundPosWeights(
    TConstArrayRef<float> targets,
    const TVector<size_t>& order
) const {
    size_t count = targets.size();
    TVector<double> posWeights(count);
    size_t queryTopSize = GetQueryTopSize(count);
    Y_ASSERT(TargetMetric == ELossFunction::PFound);
    posWeights[0] = 1.0;
    for (size_t pos = 1; pos < queryTopSize; ++pos) {
        posWeights[pos] = posWeights[pos - 1] * Decay * (1 - targets[order[pos - 1]]);
    }
    return posWeights;
}

TVector<double> TStochasticRankError::ComputeERRPosWeights(
    TConstArrayRef<float> targets,
    const TVector<size_t>& order
) const {
    size_t count = targets.size();
    TVector<double> posWeights(count);
    size_t queryTopSize = GetQueryTopSize(count);
    Y_ASSERT(TargetMetric == ELossFunction::ERR);
    posWeights[0] = 1.0;
    double pLook = 1.0;
    for (size_t pos = 1; pos < queryTopSize; ++pos) {
        pLook *= (1 - targets[order[pos - 1]]);
        posWeights[pos] = pLook / (pos + 1);
    }
    return posWeights;
}

double TStochasticRankError::CalcDCG(const TVector<float>& sortedTargets, const TVector<double>& posWeights) const {
    const size_t queryTopSize = GetQueryTopSize(sortedTargets.size());
    double result = 0.0;
    for (size_t pos = 0; pos < queryTopSize; ++pos) {
        result += CalcNumerator(sortedTargets[pos]) * posWeights[pos];
    }
    return result;
}
