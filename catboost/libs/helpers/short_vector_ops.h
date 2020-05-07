#pragma once


#include <catboost/private/libs/algo_helpers/online_predictor.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <library/cpp/sse/sse.h>

#include <util/generic/xrange.h>

#include <array>
#include <type_traits>


namespace NSimdOps {
    constexpr size_t Size = 2;
    using TValueType = double;
}

namespace NGenericSimdOps {
    using TValues = std::array<NSimdOps::TValueType, NSimdOps::Size>;

    inline TValues MakeZeros() {
        TValues result;
        result.fill(0);
        return result;
    }

    inline NSimdOps::TValueType HorizontalAdd(TValues x) {
        NSimdOps::TValueType sum = 0;
        for (auto value : x) {
            sum += value;
        }
        return sum;
    }

    inline TValues FusedMultiplyAdd(const NSimdOps::TValueType* x, const NSimdOps::TValueType* y, TValues z) {
        TValues result = z;
        for (size_t index : xrange(NSimdOps::Size)) {
            result[index] += x[index] * y[index];
        }
        return result;
    }

    inline TValues Gather(const NSimdOps::TValueType* first, const NSimdOps::TValueType* second) {
        TValues result;
        result.fill(0);
        result[0] = *first;
        result[1] = *second;
        return result;
    }

    inline TValues ElementwiseAdd(TValues x, TValues y) {
        TValues result = x;
        for (size_t index : xrange(NSimdOps::Size)) {
            result[index] += y[index];
        }
        return result;
    }

    inline void UpdateScoreBinKernelPlain(
        double scaledL2Regularizer,
        const NSimdOps::TValueType* trueStatsPtr,
        const NSimdOps::TValueType* falseStatsPtr,
        NSimdOps::TValueType* scoreBinPtr
    ) {
        const double trueAvrg = CalcAverage(
            trueStatsPtr[0],
            trueStatsPtr[1],
            scaledL2Regularizer
        );
        const double falseAvrg = CalcAverage(
            falseStatsPtr[0],
            falseStatsPtr[1],
            scaledL2Regularizer
        );
        scoreBinPtr[0] += trueAvrg * trueStatsPtr[0];
        scoreBinPtr[1] += trueAvrg * trueAvrg * trueStatsPtr[1];
        scoreBinPtr[0] += falseAvrg * falseStatsPtr[0];
        scoreBinPtr[1] += falseAvrg * falseAvrg * falseStatsPtr[1];
    }

    inline void UpdateScoreBinKernelOrdered(
        double scaledL2Regularizer,
        const NSimdOps::TValueType* trueStatsPtr,
        const NSimdOps::TValueType* falseStatsPtr,
        NSimdOps::TValueType* scoreBinPtr
    ) {
        const double trueAvrg = CalcAverage(
            trueStatsPtr[2],
            trueStatsPtr[3],
            scaledL2Regularizer
        );
        const double falseAvrg = CalcAverage(
            falseStatsPtr[2],
            falseStatsPtr[3],
            scaledL2Regularizer
        );
        scoreBinPtr[0] += trueAvrg * trueStatsPtr[0];
        scoreBinPtr[1] += trueAvrg * trueAvrg * trueStatsPtr[1];
        scoreBinPtr[0] += falseAvrg * falseStatsPtr[0];
        scoreBinPtr[1] += falseAvrg * falseAvrg * falseStatsPtr[1];
    }
}

#ifdef ARCADIA_SSE
namespace NSse2SimdOps {
    static_assert(std::is_same<NSimdOps::TValueType, double>::value, "NSimdOps::TValueType must be double");
    static_assert(NSimdOps::Size == 2, "NSimdOps::Size must be 2");
    using TValues = __m128d;

    inline TValues MakeZeros() {
        return _mm_setzero_pd();
    }

    inline NSimdOps::TValueType HorizontalAdd(TValues x) {
        return _mm_cvtsd_f64(_mm_add_pd(x, _mm_shuffle_pd(x, x, /*swap halves*/ 0x1)));
    }

    inline TValues FusedMultiplyAdd(const NSimdOps::TValueType* x, const NSimdOps::TValueType* y, TValues z) {
        return _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(x), _mm_loadu_pd(y)), z);
    }

    inline TValues Gather(const NSimdOps::TValueType* first, const NSimdOps::TValueType* second) {
        return _mm_loadh_pd(_mm_loadl_pd(_mm_undefined_pd(), first), second);
    }

    inline TValues ElementwiseAdd(TValues x, TValues y) {
        return _mm_add_pd(x, y);
    }
}
#endif

#ifdef ARCADIA_SSE
namespace NSimdOps {
    using NSse2SimdOps::MakeZeros;
    using NSse2SimdOps::HorizontalAdd;
    using NSse2SimdOps::FusedMultiplyAdd;
    using NSse2SimdOps::Gather;
    using NSse2SimdOps::ElementwiseAdd;
}
#else
namespace NSimdOps {
    using NGenericSimdOps::MakeZeros;
    using NGenericSimdOps::HorizontalAdd;
    using NGenericSimdOps::FusedMultiplyAdd;
    using NGenericSimdOps::Gather;
    using NGenericSimdOps::ElementwiseAdd;
}
#endif


#ifdef _sse_
namespace NSse2SimdOps {
    inline void UpdateScoreBinKernelPlain(
        double scaledL2Regularizer,
        const NSimdOps::TValueType* trueStatsPtr,
        const NSimdOps::TValueType* falseStatsPtr,
        NSimdOps::TValueType* scoreBinPtr
    ) {
        const __m128d trueStats = _mm_loadu_pd(trueStatsPtr);
        const __m128d falseStats = _mm_loadu_pd(falseStatsPtr);
        const __m128d prevScore = _mm_loadu_pd(scoreBinPtr);
        const __m128d sumWeightedDelta = _mm_unpacklo_pd(falseStats, trueStats);
        const __m128d sumWeight = _mm_unpackhi_pd(falseStats, trueStats);
        const __m128d regularizer = _mm_set1_pd(scaledL2Regularizer);
        const __m128d isSumWeightPositive = _mm_cmpgt_pd(sumWeight, _mm_setzero_pd());
        const __m128d average = _mm_mul_pd(sumWeightedDelta, _mm_and_pd(isSumWeightPositive, _mm_div_pd(_mm_set1_pd(1.0), _mm_add_pd(sumWeight, regularizer))));
        const __m128d dpSummands = _mm_mul_pd(average, sumWeightedDelta);
        const __m128d d2Summands = _mm_mul_pd(_mm_mul_pd(average, average), sumWeight);
        const __m128d d2DpFalse = _mm_unpacklo_pd(dpSummands, d2Summands);
        const __m128d d2DpTrue = _mm_unpackhi_pd(dpSummands, d2Summands);
        const __m128d score = _mm_add_pd(d2DpFalse, d2DpTrue);
        _mm_storeu_pd(scoreBinPtr, _mm_add_pd(prevScore, score));
    }

    inline void UpdateScoreBinKernelOrdered(
        double scaledL2Regularizer,
        const NSimdOps::TValueType* trueStatsPtr,
        const NSimdOps::TValueType* falseStatsPtr,
        NSimdOps::TValueType* scoreBinPtr
    ) {
        const __m128d trueStats = _mm_loadu_pd(trueStatsPtr + 2);
        const __m128d falseStats = _mm_loadu_pd(falseStatsPtr + 2);
        const __m128d prevScore = _mm_loadu_pd(scoreBinPtr);
        const __m128d sumDelta = _mm_unpacklo_pd(falseStats, trueStats);
        const __m128d count = _mm_unpackhi_pd(falseStats, trueStats);
        const __m128d regularizer = _mm_set1_pd(scaledL2Regularizer);
        const __m128d isCountPositive = _mm_cmpgt_pd(count, _mm_setzero_pd());
        const __m128d average = _mm_mul_pd(sumDelta, _mm_and_pd(isCountPositive, _mm_div_pd(_mm_set1_pd(1.0), _mm_add_pd(count, regularizer))));
        const __m128d trueStatsWeight = _mm_loadu_pd(trueStatsPtr);
        const __m128d falseStatsWeight = _mm_loadu_pd(falseStatsPtr);
        const __m128d sumWeightedDelta = _mm_unpacklo_pd(falseStatsWeight, trueStatsWeight);
        const __m128d sumWeight = _mm_unpackhi_pd(falseStatsWeight, trueStatsWeight);
        const __m128d dpSummands = _mm_mul_pd(average, sumWeightedDelta);
        const __m128d d2Summands = _mm_mul_pd(_mm_mul_pd(average, average), sumWeight);
        const __m128d d2DpFalse = _mm_unpacklo_pd(dpSummands, d2Summands);
        const __m128d d2DpTrue = _mm_unpackhi_pd(dpSummands, d2Summands);
        const __m128d score = _mm_add_pd(d2DpFalse, d2DpTrue);
        _mm_storeu_pd(scoreBinPtr, _mm_add_pd(prevScore, score));
    }
}
#endif

#ifdef _sse_
namespace NSimdOps {
    using NSse2SimdOps::UpdateScoreBinKernelPlain;
    using NSse2SimdOps::UpdateScoreBinKernelOrdered;
}
#else
namespace NSimdOps {
    using NGenericSimdOps::UpdateScoreBinKernelPlain;
    using NGenericSimdOps::UpdateScoreBinKernelOrdered;
}
#endif

namespace NMixedSimdOps {
#ifdef _sse3_
    inline __m128 VectorFastLogf(__m128d x0, __m128d x2) {
        const __m128i x4 = _mm_castps_si128(_mm_shuffle_ps(_mm_cvtpd_ps(x0), _mm_cvtpd_ps(x2), 0x44));
        const __m128 i4 = _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(x4, _mm_set1_epi32(0x007fffff)), _mm_set1_epi32(0x3f000000)));
        const __m128 y4 = _mm_mul_ps(_mm_cvtepi32_ps(x4), _mm_set1_ps(1.1920928955078125e-7f));
        const __m128 log4 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(y4, _mm_set1_ps(124.22551499f)), _mm_mul_ps(i4, _mm_set1_ps(1.498030302f))), _mm_div_ps(_mm_set1_ps(1.72587999f), _mm_add_ps(i4, _mm_set1_ps(0.3520887068f))));
        return _mm_mul_ps(log4, _mm_set1_ps(0.69314718f));
    }
#endif

    template <typename TIsExpApprox, typename THasDelta, typename THasWeight, typename TIsLogloss>
    inline TMetricHolder EvalCrossEntropyVectorized(
        TIsExpApprox isExpApprox,
        THasDelta hasDelta,
        THasWeight hasWeight,
        TIsLogloss isLogloss,
        TConstArrayRef<double> approx,
        TConstArrayRef<double> approxDelta,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        float border,
        int begin,
        int end,
        int* tailBegin
    ) {
        TMetricHolder result(2);
        *tailBegin = begin;
#ifndef _sse3_
        Y_UNUSED(isExpApprox);
        Y_UNUSED(hasDelta);
        Y_UNUSED(hasWeight);
        Y_UNUSED(isLogloss);
        Y_UNUSED(approx);
        Y_UNUSED(approxDelta);
        Y_UNUSED(target);
        Y_UNUSED(weight);
        Y_UNUSED(border);
        Y_UNUSED(end);
        return result;
#else
        if (!isExpApprox) {
            return result;
        }
        __m128 stat0 = _mm_setzero_ps();
        __m128 stat1 = _mm_setzero_ps();
        int idx = begin;
        for (; idx + 4 <= end; idx += 4) {
            __m128 prob = _mm_undefined_ps();
            if (isLogloss) {
                prob = _mm_and_ps(_mm_cmpgt_ps(_mm_loadu_ps(&target[idx]), _mm_set1_ps(border)), _mm_set1_ps(1.0f));
            } else {
                prob = _mm_loadu_ps(&target[idx]);
            }
            __m128d expApprox0 = _mm_loadu_pd(&approx[idx + 0]);
            __m128d expApprox2 = _mm_loadu_pd(&approx[idx + 2]);
            __m128 nonExpApprox = VectorFastLogf(expApprox0, expApprox2);
            if (hasDelta) {
                const __m128d approxDelta0 = _mm_loadu_pd(&approxDelta[idx + 0]);
                const __m128d approxDelta2 = _mm_loadu_pd(&approxDelta[idx + 2]);
                expApprox0 = _mm_mul_pd(expApprox0, approxDelta0);
                expApprox2 = _mm_mul_pd(expApprox2, approxDelta2);
                const __m128 nonExpApproxDelta = VectorFastLogf(approxDelta0, approxDelta2);
                nonExpApprox = _mm_add_ps(nonExpApprox, nonExpApproxDelta);
            }
            const __m128 isNotFinite0 = _mm_castpd_ps(_mm_cmpgt_pd(expApprox0, _mm_set1_pd(std::numeric_limits<double>::max())));
            const __m128 isNotFinite2 = _mm_castpd_ps(_mm_cmpgt_pd(expApprox2, _mm_set1_pd(std::numeric_limits<double>::max())));
            const __m128 isNotFinite = _mm_shuffle_ps(isNotFinite0, isNotFinite2, 0x88);
            const __m128 nonExpApprox1 = VectorFastLogf(
                _mm_add_pd(expApprox0, _mm_set1_pd(1.0)),
                _mm_add_pd(expApprox2, _mm_set1_pd(1.0)));
            const __m128 probTimesNonExp = _mm_mul_ps(prob, nonExpApprox);
            const __m128 nonExp1MinusProbNonExp = _mm_andnot_ps(isNotFinite, _mm_sub_ps(nonExpApprox1, probTimesNonExp));
            const __m128 negProbTimesNonExp = _mm_and_ps(isNotFinite, _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(1.0f), prob), nonExpApprox));
            __m128 w = _mm_undefined_ps();
            if (hasWeight) {
                w = _mm_loadu_ps(&weight[idx]);
            } else {
                w = _mm_set1_ps(1.0f);
            }
            stat0 = _mm_add_ps(stat0, _mm_mul_ps(w, _mm_or_ps(nonExp1MinusProbNonExp, negProbTimesNonExp)));
            stat1 = _mm_add_ps(stat1, w);
        }
        stat0 = _mm_hadd_ps(stat0, stat0);
        stat0 = _mm_hadd_ps(stat0, stat0);
        result.Stats[0] += _mm_cvtss_f32(stat0);
        stat1 = _mm_hadd_ps(stat1, stat1);
        stat1 = _mm_hadd_ps(stat1, stat1);
        result.Stats[1] += _mm_cvtss_f32(stat1);
        *tailBegin = idx;
        return result;
#endif
    }
}
