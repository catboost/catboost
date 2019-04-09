#pragma once

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
}

#ifdef _sse2_
#include <emmintrin.h>
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

#ifdef _sse2_
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
