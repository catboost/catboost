#include "dot_product.h"

#include <util/system/platform.h>

#ifdef _sse_
#include <xmmintrin.h>
#include <emmintrin.h>
#endif

#ifdef _sse_
i32 DotProduct(const i8* lhs, const i8* rhs, int length) noexcept {
    const __m128i zero = _mm_setzero_si128();
    __m128i resVec = zero;
    while (length >= 16) {
        __m128i lVec = _mm_loadu_si128((const __m128i*)lhs);
        __m128i rVec = _mm_loadu_si128((const __m128i*)rhs);

        __m128i lLo = _mm_srai_epi16(_mm_unpacklo_epi8(zero, lVec), 8);
        __m128i rLo = _mm_srai_epi16(_mm_unpacklo_epi8(zero, rVec), 8);
        __m128i lHi = _mm_srai_epi16(_mm_unpackhi_epi8(zero, lVec), 8);
        __m128i rHi = _mm_srai_epi16(_mm_unpackhi_epi8(zero, rVec), 8);

        resVec = _mm_add_epi32(resVec,
                               _mm_add_epi32(_mm_madd_epi16(lLo, rLo), _mm_madd_epi16(lHi, rHi)));

        lhs += 16;
        rhs += 16;
        length -= 16;
    }

    alignas(16) i32 res[4];
    _mm_store_si128((__m128i*)res, resVec);
    i32 sum = res[0] + res[1] + res[2] + res[3];
    for (int i = 0; i < length; ++i) {
        sum += static_cast<i32>(lhs[i]) * static_cast<i32>(rhs[i]);
    }

    return sum;
}

float DotProduct(const float* lhs, const float* rhs, int length) noexcept {
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    __m128 a1, b1, a2, b2, m1, m2;

    while (length >= 8) {
        a1 = _mm_loadu_ps(lhs);
        b1 = _mm_loadu_ps(rhs);
        m1 = _mm_mul_ps(a1, b1);

        a2 = _mm_loadu_ps(lhs + 4);
        sum1 = _mm_add_ps(sum1, m1);

        b2 = _mm_loadu_ps(rhs + 4);
        m2 = _mm_mul_ps(a2, b2);

        sum2 = _mm_add_ps(sum2, m2);

        length -= 8;
        lhs += 8;
        rhs += 8;
    }

    if (length >= 4) {
        a1 = _mm_loadu_ps(lhs);
        b1 = _mm_loadu_ps(rhs);
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(a1, b1));

        length -= 4;
        lhs += 4;
        rhs += 4;
    }

    sum1 = _mm_add_ps(sum1, sum2);

    if (length) {
        switch (length) {
            case 3:
                a1 = _mm_set_ps(lhs[0], lhs[1], lhs[2], 0.0f);
                b1 = _mm_set_ps(rhs[0], rhs[1], rhs[2], 0.0f);
                break;

            case 2:
                a1 = _mm_set_ps(lhs[0], lhs[1], 0.0f, 0.0f);
                b1 = _mm_set_ps(rhs[0], rhs[1], 0.0f, 0.0f);
                break;

            case 1:
                a1 = _mm_set_ps(lhs[0], 0.0f, 0.0f, 0.0f);
                b1 = _mm_set_ps(rhs[0], 0.0f, 0.0f, 0.0f);
                break;

            default:
                // unreachable
                a1 = _mm_setzero_ps();
                b1 = _mm_setzero_ps();
                break;
        }

        sum1 = _mm_add_ps(sum1, _mm_mul_ps(a1, b1));
    }

    alignas(16) float res[4];
    _mm_store_ps(res, sum1);

    return res[0] + res[1] + res[2] + res[3];
}

float L2NormSquared(const float* v, int length) noexcept {
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    __m128 a1, a2, m1, m2;

    while (length >= 8) {
        a1 = _mm_loadu_ps(v);
        m1 = _mm_mul_ps(a1, a1);

        a2 = _mm_loadu_ps(v + 4);
        sum1 = _mm_add_ps(sum1, m1);

        m2 = _mm_mul_ps(a2, a2);
        sum2 = _mm_add_ps(sum2, m2);

        length -= 8;
        v += 8;
    }

    if (length >= 4) {
        a1 = _mm_loadu_ps(v);
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(a1, a1));

        length -= 4;
        v += 4;
    }

    sum1 = _mm_add_ps(sum1, sum2);

    if (length) {
        switch (length) {
            case 3:
                a1 = _mm_set_ps(v[0], v[1], v[2], 0.0f);
                break;

            case 2:
                a1 = _mm_set_ps(v[0], v[1], 0.0f, 0.0f);
                break;

            case 1:
                a1 = _mm_set_ps(v[0], 0.0f, 0.0f, 0.0f);
                break;

            default:
                // unreachable
                a1 = _mm_setzero_ps();
                break;
        }

        sum1 = _mm_add_ps(sum1, _mm_mul_ps(a1, a1));
    }

    alignas(16) float res[4];
    _mm_store_ps(res, sum1);

    return res[0] + res[1] + res[2] + res[3];
}

double DotProduct(const double* lhs, const double* rhs, int length) noexcept {
    __m128d sum1 = _mm_setzero_pd();
    __m128d sum2 = _mm_setzero_pd();
    __m128d a1, b1, a2, b2;

    while (length >= 4) {
        a1 = _mm_loadu_pd(lhs);
        b1 = _mm_loadu_pd(rhs);
        sum1 = _mm_add_pd(sum1, _mm_mul_pd(a1, b1));

        a2 = _mm_loadu_pd(lhs + 2);
        b2 = _mm_loadu_pd(rhs + 2);
        sum2 = _mm_add_pd(sum2, _mm_mul_pd(a2, b2));

        length -= 4;
        lhs += 4;
        rhs += 4;
    }

    if (length >= 2) {
        a1 = _mm_loadu_pd(lhs);
        b1 = _mm_loadu_pd(rhs);
        sum1 = _mm_add_pd(sum1, _mm_mul_pd(a1, b1));

        length -= 2;
        lhs += 2;
        rhs += 2;
    }

    sum1 = _mm_add_pd(sum1, sum2);

    if (length > 0) {
        a1 = _mm_set_pd(lhs[0], 0.0);
        b1 = _mm_set_pd(rhs[0], 0.0);
        sum1 = _mm_add_pd(sum1, _mm_mul_pd(a1, b1));
    }

    alignas(16) double res[2];
    _mm_store_pd(res, sum1);

    return res[0] + res[1];
}

#else

i32 DotProduct(const i8* lhs, const i8* rhs, int length) noexcept {
    return DotProductSlow(lhs, rhs, length);
}

float DotProduct(const float* lhs, const float* rhs, int length) noexcept {
    return DotProductSlow(lhs, rhs, length);
}

double DotProduct(const double* lhs, const double* rhs, int length) noexcept {
    return DotProductSlow(lhs, rhs, length);
}

#endif // _sse_

i64 DotProduct(const i32* lhs, const i32* rhs, int length) noexcept {
    /*
     * Unfortunately there is no way of 32-bit signed integer multiplication with SSE. At least I couldn't find the way.
     * So if there is somebody who knows everithing about SSE, you are welcome.
     *
     * But this method allows processor to use vectorization and works resonably fast. You can find benchmark results
     * running bench and it is about 75% of SSE-powered speed for i8 and should be even better for i32.
     */
    i64 s0 = 0;
    i64 s1 = 0;
    i64 s2 = 0;
    i64 s3 = 0;

    while (length >= 4) {
        s0 += static_cast<i64>(lhs[0]) * static_cast<i64>(rhs[0]);
        s1 += static_cast<i64>(lhs[1]) * static_cast<i64>(rhs[1]);
        s2 += static_cast<i64>(lhs[2]) * static_cast<i64>(rhs[2]);
        s3 += static_cast<i64>(lhs[3]) * static_cast<i64>(rhs[3]);
        lhs += 4;
        rhs += 4;
        length -= 4;
    }

    while (length--)
        s0 += static_cast<i64>(*lhs++) * static_cast<i64>(*rhs++);

    return s0 + s1 + s2 + s3;
}

i32 DotProductSlow(const i8* lhs, const i8* rhs, int length) noexcept {
    i32 s0 = 0;
    i32 s1 = 0;
    i32 s2 = 0;
    i32 s3 = 0;
    i32 s4 = 0;
    i32 s5 = 0;
    i32 s6 = 0;
    i32 s7 = 0;
    i32 s8 = 0;
    i32 s9 = 0;
    i32 s10 = 0;
    i32 s11 = 0;
    i32 s12 = 0;
    i32 s13 = 0;
    i32 s14 = 0;
    i32 s15 = 0;

    while (length >= 16) {
        s0 += static_cast<i32>(lhs[0]) * static_cast<i32>(rhs[0]);
        s1 += static_cast<i32>(lhs[1]) * static_cast<i32>(rhs[1]);
        s2 += static_cast<i32>(lhs[2]) * static_cast<i32>(rhs[2]);
        s3 += static_cast<i32>(lhs[3]) * static_cast<i32>(rhs[3]);
        s4 += static_cast<i32>(lhs[4]) * static_cast<i32>(rhs[4]);
        s5 += static_cast<i32>(lhs[5]) * static_cast<i32>(rhs[5]);
        s6 += static_cast<i32>(lhs[6]) * static_cast<i32>(rhs[6]);
        s7 += static_cast<i32>(lhs[7]) * static_cast<i32>(rhs[7]);
        s8 += static_cast<i32>(lhs[8]) * static_cast<i32>(rhs[8]);
        s9 += static_cast<i32>(lhs[9]) * static_cast<i32>(rhs[9]);
        s10 += static_cast<i32>(lhs[10]) * static_cast<i32>(rhs[10]);
        s11 += static_cast<i32>(lhs[11]) * static_cast<i32>(rhs[11]);
        s12 += static_cast<i32>(lhs[12]) * static_cast<i32>(rhs[12]);
        s13 += static_cast<i32>(lhs[13]) * static_cast<i32>(rhs[13]);
        s14 += static_cast<i32>(lhs[14]) * static_cast<i32>(rhs[14]);
        s15 += static_cast<i32>(lhs[15]) * static_cast<i32>(rhs[15]);
        lhs += 16;
        rhs += 16;
        length -= 16;
    }

    while (length) {
        s0 += static_cast<i32>(*lhs++) * static_cast<i32>(*rhs++);
        --length;
    }

    return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12 + s13 + s14 + s15;
}

template <typename Res, typename Number>
static Res DotProductSlowImpl(const Number* lhs, const Number* rhs, int length) noexcept {
    Res s0 = 0;
    Res s1 = 0;
    Res s2 = 0;
    Res s3 = 0;

    while (length >= 4) {
        s0 += static_cast<Res>(lhs[0]) * static_cast<Res>(rhs[0]);
        s1 += static_cast<Res>(lhs[1]) * static_cast<Res>(rhs[1]);
        s2 += static_cast<Res>(lhs[2]) * static_cast<Res>(rhs[2]);
        s3 += static_cast<Res>(lhs[3]) * static_cast<Res>(rhs[3]);
        lhs += 4;
        rhs += 4;
        length -= 4;
    }

    while (length) {
        s0 += static_cast<Res>(*lhs++) * static_cast<Res>(*rhs++);
        --length;
    }

    return s0 + s1 + s2 + s3;
}

i64 DotProductSlow(const i32* lhs, const i32* rhs, int length) noexcept {
    return DotProductSlowImpl<i64, i32>(lhs, rhs, length);
}

float DotProductSlow(const float* lhs, const float* rhs, int length) noexcept {
    return DotProductSlowImpl<float, float>(lhs, rhs, length);
}

double DotProductSlow(const double* lhs, const double* rhs, int length) noexcept {
    return DotProductSlowImpl<double, double>(lhs, rhs, length);
}
