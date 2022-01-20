#include "dot_product_sse.h"

#include <library/cpp/sse/sse.h>
#include <util/system/platform.h>
#include <util/system/compiler.h>

#ifdef ARCADIA_SSE
i32 DotProductSse(const i8* lhs, const i8* rhs, size_t length) noexcept {
    const __m128i zero = _mm_setzero_si128();
    __m128i resVec = zero;
    while (length >= 16) {
        __m128i lVec = _mm_loadu_si128((const __m128i*)lhs);
        __m128i rVec = _mm_loadu_si128((const __m128i*)rhs);

#ifdef _sse4_1_
        __m128i lLo = _mm_cvtepi8_epi16(lVec);
        __m128i rLo = _mm_cvtepi8_epi16(rVec);
        __m128i lHi = _mm_cvtepi8_epi16(_mm_alignr_epi8(lVec, lVec, 8));
        __m128i rHi = _mm_cvtepi8_epi16(_mm_alignr_epi8(rVec, rVec, 8));
#else
        __m128i lLo = _mm_srai_epi16(_mm_unpacklo_epi8(zero, lVec), 8);
        __m128i rLo = _mm_srai_epi16(_mm_unpacklo_epi8(zero, rVec), 8);
        __m128i lHi = _mm_srai_epi16(_mm_unpackhi_epi8(zero, lVec), 8);
        __m128i rHi = _mm_srai_epi16(_mm_unpackhi_epi8(zero, rVec), 8);
#endif
        resVec = _mm_add_epi32(resVec,
                               _mm_add_epi32(_mm_madd_epi16(lLo, rLo), _mm_madd_epi16(lHi, rHi)));

        lhs += 16;
        rhs += 16;
        length -= 16;
    }

    alignas(16) i32 res[4];
    _mm_store_si128((__m128i*)res, resVec);
    i32 sum = res[0] + res[1] + res[2] + res[3];
    for (size_t i = 0; i < length; ++i) {
        sum += static_cast<i32>(lhs[i]) * static_cast<i32>(rhs[i]);
    }

    return sum;
}

ui32 DotProductSse(const ui8* lhs, const ui8* rhs, size_t length) noexcept {
    const __m128i zero = _mm_setzero_si128();
    __m128i resVec = zero;
    while (length >= 16) {
        __m128i lVec = _mm_loadu_si128((const __m128i*)lhs);
        __m128i rVec = _mm_loadu_si128((const __m128i*)rhs);

        __m128i lLo = _mm_unpacklo_epi8(lVec, zero);
        __m128i rLo = _mm_unpacklo_epi8(rVec, zero);
        __m128i lHi = _mm_unpackhi_epi8(lVec, zero);
        __m128i rHi = _mm_unpackhi_epi8(rVec, zero);

        resVec = _mm_add_epi32(resVec,
                               _mm_add_epi32(_mm_madd_epi16(lLo, rLo), _mm_madd_epi16(lHi, rHi)));

        lhs += 16;
        rhs += 16;
        length -= 16;
    }

    alignas(16) i32 res[4];
    _mm_store_si128((__m128i*)res, resVec);
    i32 sum = res[0] + res[1] + res[2] + res[3];
    for (size_t i = 0; i < length; ++i) {
        sum += static_cast<i32>(lhs[i]) * static_cast<i32>(rhs[i]);
    }

    return static_cast<ui32>(sum);
}
#ifdef _sse4_1_

i64 DotProductSse(const i32* lhs, const i32* rhs, size_t length) noexcept {
    __m128i zero = _mm_setzero_si128();
    __m128i res = zero;

    while (length >= 4) {
        __m128i a = _mm_loadu_si128((const __m128i*)lhs);
        __m128i b = _mm_loadu_si128((const __m128i*)rhs);
        res = _mm_add_epi64(_mm_mul_epi32(a, b), res);    // This is lower parts multiplication
        a = _mm_alignr_epi8(a, a, 4);
        b = _mm_alignr_epi8(b, b, 4);
        res = _mm_add_epi64(_mm_mul_epi32(a, b), res);
        rhs += 4;
        lhs += 4;
        length -= 4;
    }

    alignas(16) i64 r[2];
    _mm_store_si128((__m128i*)r, res);
    i64 sum = r[0] + r[1];

    for (size_t i = 0; i < length; ++i) {
        sum += static_cast<i64>(lhs[i]) * static_cast<i64>(rhs[i]);
    }

    return sum;
}

#else
#include "dot_product_simple.h"

i64 DotProductSse(const i32* lhs, const i32* rhs, size_t length) noexcept {
    return DotProductSimple(lhs, rhs, length);
}

#endif

float DotProductSse(const float* lhs, const float* rhs, size_t length) noexcept {
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
                a1 = _mm_set_ps(0.0f, lhs[2], lhs[1], lhs[0]);
                b1 = _mm_set_ps(0.0f, rhs[2], rhs[1], rhs[0]);
                break;

            case 2:
                a1 = _mm_set_ps(0.0f, 0.0f, lhs[1], lhs[0]);
                b1 = _mm_set_ps(0.0f, 0.0f, rhs[1], rhs[0]);
                break;

            case 1:
                a1 = _mm_set_ps(0.0f, 0.0f, 0.0f, lhs[0]);
                b1 = _mm_set_ps(0.0f, 0.0f, 0.0f, rhs[0]);
                break;

            default:
                Y_UNREACHABLE();
        }

        sum1 = _mm_add_ps(sum1, _mm_mul_ps(a1, b1));
    }

    alignas(16) float res[4];
    _mm_store_ps(res, sum1);

    return res[0] + res[1] + res[2] + res[3];
}

double DotProductSse(const double* lhs, const double* rhs, size_t length) noexcept {
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

#endif // ARCADIA_SSE
