#pragma once

#include "online_predictor.h"

#include <library/sse/sse.h>

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
