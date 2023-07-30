#include "l2_distance.h"

#include <library/cpp/sse/sse.h>

#include <contrib/libs/cblas/include/cblas.h>

#include <util/system/platform.h>

template <typename Result, typename Number>
inline Result SqrDelta(Number a, Number b) {
    Result diff = a < b ? b - a : a - b;
    return diff * diff;
}

template <typename Result, typename Number>
inline Result L2SqrDistanceImpl(const Number* a, const Number* b, int length) {
    Result res = 0;

    for (int i = 0; i < length; i++) {
        res += SqrDelta<Result, Number>(a[i], b[i]);
    }

    return res;
}

template <typename Result, typename Number>
inline Result L2SqrDistanceImpl2(const Number* a, const Number* b, int length) {
    Result s0 = 0;
    Result s1 = 0;

    while (length >= 2) {
        s0 += SqrDelta<Result, Number>(a[0], b[0]);
        s1 += SqrDelta<Result, Number>(a[1], b[1]);
        a += 2;
        b += 2;
        length -= 2;
    }

    while (length--)
        s0 += SqrDelta<Result, Number>(*a++, *b++);

    return s0 + s1;
}

template <typename Result, typename Number>
inline Result L2SqrDistanceImpl4(const Number* a, const Number* b, int length) {
    Result s0 = 0;
    Result s1 = 0;
    Result s2 = 0;
    Result s3 = 0;

    while (length >= 4) {
        s0 += SqrDelta<Result, Number>(a[0], b[0]);
        s1 += SqrDelta<Result, Number>(a[1], b[1]);
        s2 += SqrDelta<Result, Number>(a[2], b[2]);
        s3 += SqrDelta<Result, Number>(a[3], b[3]);
        a += 4;
        b += 4;
        length -= 4;
    }

    while (length--)
        s0 += SqrDelta<Result, Number>(*a++, *b++);

    return s0 + s1 + s2 + s3;
}

inline ui32 L2SqrDistanceImplUI4(const ui8* a, const ui8* b, int length) {
    ui32 res = 0;
    for (int i = 0; i < length; i++) {
        res += SqrDelta<ui32, ui8>(a[i] & 0x0f, b[i] & 0x0f);
        res += SqrDelta<ui32, ui8>(a[i] & 0xf0, b[i] & 0xf0) >> 8;
    }
    return res;
}


#ifdef ARCADIA_SSE
namespace NL2Distance {
    static const __m128i MASK_UI4_1 = _mm_set_epi8(0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
                                                   0x0f, 0x0f, 0x0f, 0x0f, 0x0f);
    static const __m128i MASK_UI4_2 = _mm_set_epi8(0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
                                                   0xf0, 0xf0, 0xf0, 0xf0, 0xf0);
}
ui32 L2SqrDistance(const i8* lhs, const i8* rhs, int length) {
    const __m128i zero = _mm_setzero_si128();
    __m128i resVec = zero;

    while (length >= 16) {
        __m128i vec = _mm_subs_epi8(_mm_loadu_si128((const __m128i*)lhs), _mm_loadu_si128((const __m128i*)rhs));

#ifdef _sse4_1_
        __m128i lo = _mm_cvtepi8_epi16(vec);
        __m128i hi = _mm_cvtepi8_epi16(_mm_alignr_epi8(vec, vec, 8));
#else
        __m128i lo = _mm_srai_epi16(_mm_unpacklo_epi8(zero, vec), 8);
        __m128i hi = _mm_srai_epi16(_mm_unpackhi_epi8(zero, vec), 8);
#endif

        resVec = _mm_add_epi32(resVec,
                               _mm_add_epi32(_mm_madd_epi16(lo, lo), _mm_madd_epi16(hi, hi)));

        lhs += 16;
        rhs += 16;
        length -= 16;
    }

    alignas(16) ui32 res[4];
    _mm_store_si128((__m128i*)res, resVec);
    ui32 sum = res[0] + res[1] + res[2] + res[3];
    for (int i = 0; i < length; ++i) {
        sum += Sqr(static_cast<i32>(lhs[i]) - static_cast<i32>(rhs[i]));
    }

    return sum;
}

ui32 L2SqrDistance(const ui8* lhs, const ui8* rhs, int length) {
    const __m128i zero = _mm_setzero_si128();
    __m128i resVec = zero;

    while (length >= 16) {
        __m128i lVec = _mm_loadu_si128((const __m128i*)lhs);
        __m128i rVec = _mm_loadu_si128((const __m128i*)rhs);

        // We will think about this vectors as about i16.
        __m128i lo = _mm_sub_epi16(_mm_unpacklo_epi8(lVec, zero), _mm_unpacklo_epi8(rVec, zero));
        __m128i hi = _mm_sub_epi16(_mm_unpackhi_epi8(lVec, zero), _mm_unpackhi_epi8(rVec, zero));

        resVec = _mm_add_epi32(resVec,
                               _mm_add_epi32(_mm_madd_epi16(lo, lo), _mm_madd_epi16(hi, hi)));

        lhs += 16;
        rhs += 16;
        length -= 16;
    }

    alignas(16) ui32 res[4];
    _mm_store_si128((__m128i*)res, resVec);
    ui32 sum = res[0] + res[1] + res[2] + res[3];
    for (int i = 0; i < length; ++i) {
        sum += Sqr(static_cast<i32>(lhs[i]) - static_cast<i32>(rhs[i]));
    }

    return sum;
}

float L2SqrDistance(const float* lhs, const float* rhs, int length) {
    __m128 sum = _mm_setzero_ps();

    while (length >= 4) {
        __m128 a = _mm_loadu_ps(lhs);
        __m128 b = _mm_loadu_ps(rhs);
        __m128 delta = _mm_sub_ps(a, b);
        sum = _mm_add_ps(sum, _mm_mul_ps(delta, delta));
        length -= 4;
        rhs += 4;
        lhs += 4;
    }

    alignas(16) float res[4];
    _mm_store_ps(res, sum);

    while (length--)
        res[0] += Sqr(*rhs++ - *lhs++);

    return res[0] + res[1] + res[2] + res[3];
}

double L2SqrDistance(const double* lhs, const double* rhs, int length) {
    __m128d sum = _mm_setzero_pd();

    while (length >= 2) {
        __m128d a = _mm_loadu_pd(lhs);
        __m128d b = _mm_loadu_pd(rhs);
        __m128d delta = _mm_sub_pd(a, b);
        sum = _mm_add_pd(sum, _mm_mul_pd(delta, delta));
        length -= 2;
        rhs += 2;
        lhs += 2;
    }

    alignas(16) double res[2];
    _mm_store_pd(res, sum);

    while (length--)
        res[0] += Sqr(*rhs++ - *lhs++);

    return res[0] + res[1];
}

ui64 L2SqrDistance(const i32* lhs, const i32* rhs, int length) {
    __m128i zero = _mm_setzero_si128();
    __m128i res = zero;

    while (length >= 4) {
        __m128i a = _mm_loadu_si128((const __m128i*)lhs);
        __m128i b = _mm_loadu_si128((const __m128i*)rhs);

#ifdef _sse4_1_
        // In SSE4.1 si32*si32->si64 is available, so we may do just (a-b)*(a-b) not caring about (a-b) sign
        a = _mm_sub_epi32(a, b);
        res = _mm_add_epi64(_mm_mul_epi32(a, a), res);
        a = _mm_alignr_epi8(a, a, 4);
        res = _mm_add_epi64(_mm_mul_epi32(a, a), res);
#else
        __m128i mask = _mm_cmpgt_epi32(a, b);                                       // mask = a > b? 0xffffffff: 0;
        __m128i a2 = _mm_sub_epi32(_mm_and_si128(mask, a), _mm_and_si128(mask, b)); // a2 = (a & mask) - (b & mask) (for a > b)
        b = _mm_sub_epi32(_mm_andnot_si128(mask, b), _mm_andnot_si128(mask, a));    // b = (b & ~mask) - (a & ~mask)   (for b > a)
        a = _mm_or_si128(a2, b);                                                    // a = abs(a - b)
        a2 = _mm_unpackhi_epi32(a, zero);
        res = _mm_add_epi64(_mm_mul_epu32(a2, a2), res);
        a2 = _mm_unpacklo_epi32(a, zero);
        res = _mm_add_epi64(_mm_mul_epu32(a2, a2), res);
#endif

        rhs += 4;
        lhs += 4;
        length -= 4;
    }

    alignas(16) ui64 r[2];
    _mm_store_si128((__m128i*)r, res);
    ui64 sum = r[0] + r[1];

    while (length) {
        sum += SqrDelta<ui64, i32>(lhs[0], rhs[0]);
        ++lhs;
        ++rhs;
        --length;
    }

    return sum;
}

ui64 L2SqrDistance(const ui32* lhs, const ui32* rhs, int length) {
    __m128i zero = _mm_setzero_si128();
    __m128i shift = _mm_set1_epi32(0x80000000);
    __m128i res = zero;

    while (length >= 4) {
        __m128i a = _mm_add_epi32(_mm_loadu_si128((const __m128i*)lhs), shift);
        __m128i b = _mm_add_epi32(_mm_loadu_si128((const __m128i*)rhs), shift);
        __m128i mask = _mm_cmpgt_epi32(a, b);                                       // mask = a > b? 0xffffffff: 0;
        __m128i a2 = _mm_sub_epi32(_mm_and_si128(mask, a), _mm_and_si128(mask, b)); // a2 = (a & mask) - (b & mask) (for a > b)
        b = _mm_sub_epi32(_mm_andnot_si128(mask, b), _mm_andnot_si128(mask, a));    // b = (b & ~mask) - (a & ~mask)   (for b > a)
        a = _mm_or_si128(a2, b);                                                    // a = abs(a - b)

#ifdef _sse4_1_
        res = _mm_add_epi64(_mm_mul_epu32(a, a), res);
        a = _mm_alignr_epi8(a, a, 4);
        res = _mm_add_epi64(_mm_mul_epu32(a, a), res);
#else
        a2 = _mm_unpackhi_epi32(a, zero);
        res = _mm_add_epi64(_mm_mul_epu32(a2, a2), res);
        a2 = _mm_unpacklo_epi32(a, zero);
        res = _mm_add_epi64(_mm_mul_epu32(a2, a2), res);
#endif

        rhs += 4;
        lhs += 4;
        length -= 4;
    }

    alignas(16) ui64 r[2];
    _mm_store_si128((__m128i*)r, res);
    ui64 sum = r[0] + r[1];

    while (length) {
        sum += SqrDelta<ui64, ui32>(lhs[0], rhs[0]);
        ++lhs;
        ++rhs;
        --length;
    }

    return sum;
}

ui32 L2SqrDistanceUI4(const ui8* lhs, const ui8* rhs, int length) {
    const __m128i zero = _mm_setzero_si128();
    __m128i resVec1 = zero;
    __m128i resVec2 = zero;

    while (length >= 16) {
        __m128i lVec = _mm_loadu_si128((const __m128i*)lhs);
        __m128i rVec = _mm_loadu_si128((const __m128i*)rhs);

        __m128i lVec1 = _mm_and_si128(lVec, NL2Distance::MASK_UI4_1);
        __m128i lVec2 = _mm_and_si128(lVec, NL2Distance::MASK_UI4_2);
        __m128i rVec1 = _mm_and_si128(rVec, NL2Distance::MASK_UI4_1);
        __m128i rVec2 = _mm_and_si128(rVec, NL2Distance::MASK_UI4_2);
        // We will think about this vectors as about i16.
        __m128i lo1 = _mm_sub_epi16(_mm_unpacklo_epi8(lVec1, zero), _mm_unpacklo_epi8(rVec1, zero));
        __m128i hi1 = _mm_sub_epi16(_mm_unpackhi_epi8(lVec1, zero), _mm_unpackhi_epi8(rVec1, zero));
        __m128i lo2 = _mm_sub_epi16(_mm_unpacklo_epi8(lVec2, zero), _mm_unpacklo_epi8(rVec2, zero));
        __m128i hi2 = _mm_sub_epi16(_mm_unpackhi_epi8(lVec2, zero), _mm_unpackhi_epi8(rVec2, zero));

        resVec1 = _mm_add_epi32(resVec1, _mm_add_epi32(_mm_madd_epi16(lo1, lo1), _mm_madd_epi16(hi1, hi1)));
        resVec2 = _mm_add_epi32(resVec2, _mm_add_epi32(_mm_madd_epi16(lo2, lo2), _mm_madd_epi16(hi2, hi2)));

        lhs += 16;
        rhs += 16;
        length -= 16;
    }

    alignas(16) ui32 res[4];
    _mm_store_si128((__m128i*)res, resVec1);
    ui32 sum = res[0] + res[1] + res[2] + res[3];
    _mm_store_si128((__m128i*)res, resVec2);
    sum += (res[0] + res[1] + res[2] + res[3]) >> 8;
    for (int i = 0; i < length; ++i) {
        sum += Sqr(static_cast<i32>(lhs[i] & 0x0f) - static_cast<i32>(rhs[i] & 0x0f));
        sum += Sqr(static_cast<i32>(lhs[i] & 0xf0) - static_cast<i32>(rhs[i] & 0xf0)) >> 8;
    }
    return sum;
}

#else /* !ARCADIA_SSE */

ui32 L2SqrDistance(const i8* lhs, const i8* rhs, int length) {
    return L2SqrDistanceImpl<ui32, i8>(lhs, rhs, length);
}

ui32 L2SqrDistance(const ui8* lhs, const ui8* rhs, int length) {
    return L2SqrDistanceImpl<ui32, ui8>(lhs, rhs, length);
}

ui64 L2SqrDistance(const i32* a, const i32* b, int length) {
    return L2SqrDistanceImpl2<ui64, i32>(a, b, length);
}

ui64 L2SqrDistance(const ui32* a, const ui32* b, int length) {
    return L2SqrDistanceImpl2<ui64, ui32>(a, b, length);
}

float L2SqrDistance(const float* a, const float* b, int length) {
    return L2SqrDistanceImpl4<float, float>(a, b, length);
}

double L2SqrDistance(const double* a, const double* b, int length) {
    return L2SqrDistanceImpl2<double, double>(a, b, length);
}

ui32 L2SqrDistanceUI4(const ui8* lhs, const ui8* rhs, int length) {
    return L2SqrDistanceImplUI4(lhs, rhs, length);
}

#endif /* ARCADIA_SSE */

ui32 L2SqrDistanceSlow(const i8* lhs, const i8* rhs, int length) {
    return L2SqrDistanceImpl<ui32, i8>(lhs, rhs, length);
}

ui32 L2SqrDistanceSlow(const ui8* lhs, const ui8* rhs, int length) {
    return L2SqrDistanceImpl<ui32, ui8>(lhs, rhs, length);
}

ui64 L2SqrDistanceSlow(const i32* a, const i32* b, int length) {
    return L2SqrDistanceImpl2<ui64, i32>(a, b, length);
}

ui64 L2SqrDistanceSlow(const ui32* a, const ui32* b, int length) {
    return L2SqrDistanceImpl2<ui64, ui32>(a, b, length);
}

float L2SqrDistanceSlow(const float* a, const float* b, int length) {
    return L2SqrDistanceImpl4<float, float>(a, b, length);
}

double L2SqrDistanceSlow(const double* a, const double* b, int length) {
    return L2SqrDistanceImpl2<double, double>(a, b, length);
}

ui32 L2SqrDistanceUI4Slow(const ui8* lhs, const ui8* rhs, int length) {
    return L2SqrDistanceImplUI4(lhs, rhs, length);
}
