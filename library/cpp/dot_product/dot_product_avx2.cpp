#include "dot_product_avx2.h"
#include "dot_product_simple.h"
#include "dot_product_sse.h"

#if defined(_avx2_) && defined(_fma_)

#include <util/system/platform.h>
#include <util/system/compiler.h>
#include <util/generic/utility.h>

#include <immintrin.h>

namespace {
    constexpr i64 Bits(int n) {
        return i64(-1) ^ ((i64(1) << (64 - n)) - 1);
    }

    constexpr __m256 BlendMask64[8] = {
        __m256i{Bits(64), Bits(64), Bits(64), Bits(64)},
        __m256i{0, Bits(64), Bits(64), Bits(64)},
        __m256i{0, 0, Bits(64), Bits(64)},
        __m256i{0, 0, 0, Bits(64)},
    };

    constexpr __m256 BlendMask32[8] = {
        __m256i{Bits(64), Bits(64), Bits(64), Bits(64)},
        __m256i{Bits(32), Bits(64), Bits(64), Bits(64)},
        __m256i{0, Bits(64), Bits(64), Bits(64)},
        __m256i{0, Bits(32), Bits(64), Bits(64)},
        __m256i{0, 0, Bits(64), Bits(64)},
        __m256i{0, 0, Bits(32), Bits(64)},
        __m256i{0, 0, 0, Bits(64)},
        __m256i{0, 0, 0, Bits(32)},
    };

    constexpr __m128 BlendMask8[16] = {
        __m128i{Bits(64), Bits(64)},
        __m128i{Bits(56), Bits(64)},
        __m128i{Bits(48), Bits(64)},
        __m128i{Bits(40), Bits(64)},
        __m128i{Bits(32), Bits(64)},
        __m128i{Bits(24), Bits(64)},
        __m128i{Bits(16), Bits(64)},
        __m128i{Bits(8), Bits(64)},
        __m128i{0, Bits(64)},
        __m128i{0, Bits(56)},
        __m128i{0, Bits(48)},
        __m128i{0, Bits(40)},
        __m128i{0, Bits(32)},
        __m128i{0, Bits(24)},
        __m128i{0, Bits(16)},
        __m128i{0, Bits(8)},
    };

    // See https://stackoverflow.com/a/60109639
    // Horizontal sum of eight i32 values in an avx register
    i32 HsumI32(__m256i v) {
        __m128i x = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        __m128i hi64  = _mm_unpackhi_epi64(x, x);
        __m128i sum64 = _mm_add_epi32(hi64, x);
        __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        __m128i sum32 = _mm_add_epi32(sum64, hi32);
        return _mm_cvtsi128_si32(sum32);
    }

    // Horizontal sum of four i64 values in an avx register
    i64 HsumI64(__m256i v) {
        __m128i x = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        return _mm_cvtsi128_si64(x) + _mm_extract_epi64(x, 1);
    }

    // Horizontal sum of eight float values in an avx register
    float HsumFloat(__m256 v) {
        __m256 y = _mm256_permute2f128_ps(v, v, 1);
        v = _mm256_add_ps(v, y);
        v = _mm256_hadd_ps(v, v);
        return _mm256_cvtss_f32(_mm256_hadd_ps(v, v));
    }

    // Horizontal sum of four double values in an avx register
    double HsumDouble(__m256 v) {
        __m128d x = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
        x = _mm_add_pd(x, _mm_shuffle_pd(x, x, 1));
        return _mm_cvtsd_f64(x);
    }

    __m128i Load128i(const void* ptr) {
        return _mm_loadu_si128((const __m128i*)ptr);
    }

    __m256i Load256i(const void* ptr) {
        return _mm256_loadu_si256((const __m256i*)ptr);
    }

    // Unrolled dot product for relatively small sizes
    // The loop with known upper bound is unrolled by the compiler, no need to do anything special about it
    template <size_t size, class TInput, class TExtend>
    i32 DotProductInt8Avx2_Unroll(const TInput* lhs, const TInput* rhs, TExtend extend) noexcept {
        static_assert(size % 16 == 0);
        auto sum = _mm256_setzero_ps();
        for (size_t i = 0; i != size; i += 16) {
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(extend(Load128i(lhs + i)), extend(Load128i(rhs + i))));
        }

        return HsumI32(sum);
    }

    template <class TInput, class TExtend>
    i32 DotProductInt8Avx2(const TInput* lhs, const TInput* rhs, size_t length, TExtend extend) noexcept {
        // Fully unrolled versions for small multiples for 16
        switch (length) {
            case 16: return DotProductInt8Avx2_Unroll<16>(lhs, rhs, extend);
            case 32: return DotProductInt8Avx2_Unroll<32>(lhs, rhs, extend);
            case 48: return DotProductInt8Avx2_Unroll<48>(lhs, rhs, extend);
            case 64: return DotProductInt8Avx2_Unroll<64>(lhs, rhs, extend);
        }

        __m256i sum = _mm256_setzero_ps();

        if (const auto leftover = length % 16; leftover != 0) {
            auto a = _mm_blendv_epi8(
                    Load128i(lhs), _mm_setzero_ps(), BlendMask8[leftover]);
            auto b = _mm_blendv_epi8(
                    Load128i(rhs), _mm_setzero_ps(), BlendMask8[leftover]);

            sum = _mm256_madd_epi16(extend(a), extend(b));

            lhs += leftover;
            rhs += leftover;
            length -= leftover;
        }

        while (length >= 32) {
            const auto l0 = extend(Load128i(lhs));
            const auto r0 = extend(Load128i(rhs));
            const auto l1 = extend(Load128i(lhs + 16));
            const auto r1 = extend(Load128i(rhs + 16));

            const auto s0 = _mm256_madd_epi16(l0, r0);
            const auto s1 = _mm256_madd_epi16(l1, r1);

            sum = _mm256_add_epi32(sum, _mm256_add_epi32(s0, s1));

            lhs += 32;
            rhs += 32;
            length -= 32;
        }

        if (length > 0) {
            auto l = extend(Load128i(lhs));
            auto r = extend(Load128i(rhs));

            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(l, r));
        }

        return HsumI32(sum);
    }
}

i32 DotProductAvx2(const i8* lhs, const i8* rhs, size_t length) noexcept {
    if (length < 16) {
        return DotProductSse(lhs, rhs, length);
    }
    return DotProductInt8Avx2(lhs, rhs, length, [](const __m128i x) {
        return _mm256_cvtepi8_epi16(x);
    });
}

ui32 DotProductAvx2(const ui8* lhs, const ui8* rhs, size_t length) noexcept {
    if (length < 16) {
        return DotProductSse(lhs, rhs, length);
    }
    return DotProductInt8Avx2(lhs, rhs, length, [](const __m128i x) {
        return _mm256_cvtepu8_epi16(x);
    });
}

i64 DotProductAvx2(const i32* lhs, const i32* rhs, size_t length) noexcept {
    if (length < 16) {
        return DotProductSse(lhs, rhs, length);
    }
    __m256i res = _mm256_setzero_ps();

    if (const auto leftover = length % 8; leftover != 0) {
        // Use floating-point blendv. Who cares as long as the size is right.
        __m256i a = _mm256_blendv_ps(
                Load256i(lhs), _mm256_setzero_ps(), BlendMask32[leftover]);
        __m256i b = _mm256_blendv_ps(
                Load256i(rhs), _mm256_setzero_ps(), BlendMask32[leftover]);

        res = _mm256_mul_epi32(a, b);
        a = _mm256_alignr_epi8(a, a, 4);
        b = _mm256_alignr_epi8(b, b, 4);
        res = _mm256_add_epi64(_mm256_mul_epi32(a, b), res);

        lhs += leftover;
        rhs += leftover;
        length -= leftover;
    }

    while (length >= 8) {
        __m256i a = Load256i(lhs);
        __m256i b = Load256i(rhs);
        res = _mm256_add_epi64(_mm256_mul_epi32(a, b), res);    // This is lower parts multiplication
        a = _mm256_alignr_epi8(a, a, 4);
        b = _mm256_alignr_epi8(b, b, 4);
        res = _mm256_add_epi64(_mm256_mul_epi32(a, b), res);
        rhs += 8;
        lhs += 8;
        length -= 8;
    }

    return HsumI64(res);
}

float DotProductAvx2(const float* lhs, const float* rhs, size_t length) noexcept {
    if (length < 16) {
        return DotProductSse(lhs, rhs, length);
    }
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 a1, b1, a2, b2;

    if (const auto leftover = length % 8; leftover != 0) {
        a1 = _mm256_blendv_ps(
                _mm256_loadu_ps(lhs), _mm256_setzero_ps(), BlendMask32[leftover]);
        b1 = _mm256_blendv_ps(
                _mm256_loadu_ps(rhs), _mm256_setzero_ps(), BlendMask32[leftover]);
        sum1 = _mm256_mul_ps(a1, b1);
        lhs += leftover;
        rhs += leftover;
        length -= leftover;
    }

    while (length >= 16) {
        a1 = _mm256_loadu_ps(lhs);
        b1 = _mm256_loadu_ps(rhs);
        a2 = _mm256_loadu_ps(lhs + 8);
        b2 = _mm256_loadu_ps(rhs + 8);

        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);

        length -= 16;
        lhs += 16;
        rhs += 16;
    }

    if (length > 0) {
        a1 = _mm256_loadu_ps(lhs);
        b1 = _mm256_loadu_ps(rhs);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
    }

    return HsumFloat(_mm256_add_ps(sum1, sum2));
}

double DotProductAvx2(const double* lhs, const double* rhs, size_t length) noexcept {
    if (length < 16) {
        return DotProductSse(lhs, rhs, length);
    }
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d a1, b1, a2, b2;

    if (const auto leftover = length % 4; leftover != 0) {
        a1 = _mm256_blendv_pd(
                _mm256_loadu_pd(lhs), _mm256_setzero_ps(), BlendMask64[leftover]);
        b1 = _mm256_blendv_pd(
                _mm256_loadu_pd(rhs), _mm256_setzero_ps(), BlendMask64[leftover]);
        sum1 = _mm256_mul_pd(a1, b1);
        lhs += leftover;
        rhs += leftover;
        length -= leftover;
    }

    while (length >= 8) {
        a1 = _mm256_loadu_pd(lhs);
        b1 = _mm256_loadu_pd(rhs);
        a2 = _mm256_loadu_pd(lhs + 4);
        b2 = _mm256_loadu_pd(rhs + 4);

        sum1 = _mm256_fmadd_pd(a1, b1, sum1);
        sum2 = _mm256_fmadd_pd(a2, b2, sum2);

        length -= 8;
        lhs += 8;
        rhs += 8;
    }

    if (length > 0) {
        a1 = _mm256_loadu_pd(lhs);
        b1 = _mm256_loadu_pd(rhs);
        sum1 = _mm256_fmadd_pd(a1, b1, sum1);
    }

    return HsumDouble(_mm256_add_pd(sum1, sum2));
}

#elif defined(ARCADIA_SSE)

i32 DotProductAvx2(const i8* lhs, const i8* rhs, size_t length) noexcept {
    return DotProductSse(lhs, rhs, length);
}

ui32 DotProductAvx2(const ui8* lhs, const ui8* rhs, size_t length) noexcept {
    return DotProductSse(lhs, rhs, length);
}

i64 DotProductAvx2(const i32* lhs, const i32* rhs, size_t length) noexcept {
    return DotProductSse(lhs, rhs, length);
}

float DotProductAvx2(const float* lhs, const float* rhs, size_t length) noexcept {
    return DotProductSse(lhs, rhs, length);
}

double DotProductAvx2(const double* lhs, const double* rhs, size_t length) noexcept {
    return DotProductSse(lhs, rhs, length);
}

#else

i32 DotProductAvx2(const i8* lhs, const i8* rhs, size_t length) noexcept {
    return DotProductSimple(lhs, rhs, length);
}

ui32 DotProductAvx2(const ui8* lhs, const ui8* rhs, size_t length) noexcept {
    return DotProductSimple(lhs, rhs, length);
}

i64 DotProductAvx2(const i32* lhs, const i32* rhs, size_t length) noexcept {
    return DotProductSimple(lhs, rhs, length);
}

float DotProductAvx2(const float* lhs, const float* rhs, size_t length) noexcept {
    return DotProductSimple(lhs, rhs, length);
}

double DotProductAvx2(const double* lhs, const double* rhs, size_t length) noexcept {
    return DotProductSimple(lhs, rhs, length);
}

#endif
