#include "dot_product.h"

#include <library/cpp/sse/sse.h>
#include <util/system/platform.h>
#include <util/system/compiler.h>
#include <util/generic/utility.h>

#ifdef ARCADIA_SSE
i32 DotProduct(const i8* lhs, const i8* rhs, ui32 length) noexcept {
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
    for (ui32 i = 0; i < length; ++i) {
        sum += static_cast<i32>(lhs[i]) * static_cast<i32>(rhs[i]);
    }

    return sum;
}

ui32 DotProduct(const ui8* lhs, const ui8* rhs, ui32 length) noexcept {
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
    for (ui32 i = 0; i < length; ++i) {
        sum += static_cast<i32>(lhs[i]) * static_cast<i32>(rhs[i]);
    }

    return static_cast<ui32>(sum);
}
#ifdef _sse4_1_

i64 DotProduct(const i32* lhs, const i32* rhs, ui32 length) noexcept {
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

    for (ui32 i = 0; i < length; ++i) {
        sum += static_cast<i64>(lhs[i]) * static_cast<i64>(rhs[i]);
    }

    return sum;
}

#else

i64 DotProduct(const i32* lhs, const i32* rhs, ui32 length) noexcept {
    return DotProductSlow(lhs, rhs, length);
}

#endif

float DotProduct(const float* lhs, const float* rhs, ui32 length) noexcept {
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

float L2NormSquared(const float* v, ui32 length) noexcept {
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
                a1 = _mm_set_ps(0.0f, v[2], v[1], v[0]);
                break;

            case 2:
                a1 = _mm_set_ps(0.0f, 0.0f, v[1], v[0]);
                break;

            case 1:
                a1 = _mm_set_ps(0.0f, 0.0f, 0.0f, v[0]);
                break;

            default:
                Y_UNREACHABLE();
        }

        sum1 = _mm_add_ps(sum1, _mm_mul_ps(a1, a1));
    }

    alignas(16) float res[4];
    _mm_store_ps(res, sum1);

    return res[0] + res[1] + res[2] + res[3];
}

double DotProduct(const double* lhs, const double* rhs, ui32 length) noexcept {
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


template <bool computeLL, bool computeLR, bool computeRR>
Y_FORCE_INLINE
static void TriWayDotProductIteration(__m128& sumLL, __m128& sumLR, __m128& sumRR, const __m128 a, const __m128 b) {
    if constexpr (computeLL) {
        sumLL = _mm_add_ps(sumLL, _mm_mul_ps(a, a));
    }
    if constexpr (computeLR) {
        sumLR = _mm_add_ps(sumLR, _mm_mul_ps(a, b));
    }
    if constexpr (computeRR) {
        sumRR = _mm_add_ps(sumRR, _mm_mul_ps(b, b));
    }
}


template <bool computeLL, bool computeLR, bool computeRR>
static TTriWayDotProduct<float> TriWayDotProductImpl(const float* lhs, const float* rhs, ui32 length) noexcept {
    __m128 sumLL1 = _mm_setzero_ps();
    __m128 sumLR1 = _mm_setzero_ps();
    __m128 sumRR1 = _mm_setzero_ps();
    __m128 sumLL2 = _mm_setzero_ps();
    __m128 sumLR2 = _mm_setzero_ps();
    __m128 sumRR2 = _mm_setzero_ps();

    while (length >= 8) {
        TriWayDotProductIteration<computeLL, computeLR, computeRR>(sumLL1, sumLR1, sumRR1, _mm_loadu_ps(lhs + 0), _mm_loadu_ps(rhs + 0));
        TriWayDotProductIteration<computeLL, computeLR, computeRR>(sumLL2, sumLR2, sumRR2, _mm_loadu_ps(lhs + 4), _mm_loadu_ps(rhs + 4));
        length -= 8;
        lhs += 8;
        rhs += 8;
    }

    if (length >= 4) {
        TriWayDotProductIteration<computeLL, computeLR, computeRR>(sumLL1, sumLR1, sumRR1, _mm_loadu_ps(lhs + 0), _mm_loadu_ps(rhs + 0));
        length -= 4;
        lhs += 4;
        rhs += 4;
    }

    if constexpr (computeLL) {
        sumLL1 = _mm_add_ps(sumLL1, sumLL2);
    }
    if constexpr (computeLR) {
        sumLR1 = _mm_add_ps(sumLR1, sumLR2);
    }
    if constexpr (computeRR) {
        sumRR1 = _mm_add_ps(sumRR1, sumRR2);
    }

    if (length) {
        __m128 a, b;
        switch (length) {
            case 3:
                a = _mm_set_ps(0.0f, lhs[2], lhs[1], lhs[0]);
                b = _mm_set_ps(0.0f, rhs[2], rhs[1], rhs[0]);
                break;
            case 2:
                a = _mm_set_ps(0.0f, 0.0f, lhs[1], lhs[0]);
                b = _mm_set_ps(0.0f, 0.0f, rhs[1], rhs[0]);
                break;
            case 1:
                a = _mm_set_ps(0.0f, 0.0f, 0.0f, lhs[0]);
                b = _mm_set_ps(0.0f, 0.0f, 0.0f, rhs[0]);
                break;
            default:
                Y_UNREACHABLE();
        }
        TriWayDotProductIteration<computeLL, computeLR, computeRR>(sumLL1, sumLR1, sumRR1, a, b);
    }

    __m128 t0 = sumLL1;
    __m128 t1 = sumLR1;
    __m128 t2 = sumRR1;
    __m128 t3 = _mm_setzero_ps();
    _MM_TRANSPOSE4_PS(t0, t1, t2, t3);
    t0 = _mm_add_ps(t0, t1);
    t0 = _mm_add_ps(t0, t2);
    t0 = _mm_add_ps(t0, t3);

    alignas(16) float res[4];
    _mm_store_ps(res, t0);
    TTriWayDotProduct<float> result{res[0], res[1], res[2]};
    static constexpr const TTriWayDotProduct<float> def;
    // fill skipped fields with default values
    if constexpr (!computeLL) {
        result.LL = def.LL;
    }
    if constexpr (!computeLR) {
        result.LR = def.LR;
    }
    if constexpr (!computeRR) {
        result.RR = def.RR;
    }
    return result;
}


TTriWayDotProduct<float> TriWayDotProduct(const float* lhs, const float* rhs, ui32 length, unsigned mask) noexcept {
    mask &= 0b111;
    if (Y_LIKELY(mask == 0b111)) { // compute dot-product and length² of two vectors
        return TriWayDotProductImpl<true, true, true>(lhs, rhs, length);
    } else if (Y_LIKELY(mask == 0b110 || mask == 0b011)) { // compute dot-product and length² of one vector
        const bool computeLL = (mask == 0b110);
        if (!computeLL) {
            DoSwap(lhs, rhs);
        }
        auto result = TriWayDotProductImpl<true, true, false>(lhs, rhs, length);
        if (!computeLL) {
            DoSwap(result.LL, result.RR);
        }
        return result;
    } else {
        // dispatch unlikely & sparse cases
        TTriWayDotProduct<float> result{};
        switch(mask) {
            case 0b000:
                break;
            case 0b100:
                result.LL = L2NormSquared(lhs, length);
                break;
            case 0b010:
                result.LR = DotProduct(lhs, rhs, length);
                break;
            case 0b001:
                result.RR = L2NormSquared(rhs, length);
                break;
            case 0b101:
                result.LL = L2NormSquared(lhs, length);
                result.RR = L2NormSquared(rhs, length);
                break;
            default:
                Y_UNREACHABLE();
        }
        return result;
    }
}

#else

i32 DotProduct(const i8* lhs, const i8* rhs, ui32 length) noexcept {
    return DotProductSlow(lhs, rhs, length);
}

float DotProduct(const float* lhs, const float* rhs, ui32 length) noexcept {
    return DotProductSlow(lhs, rhs, length);
}

double DotProduct(const double* lhs, const double* rhs, ui32 length) noexcept {
    return DotProductSlow(lhs, rhs, length);
}

float L2NormSquared(const float* v, ui32 length) noexcept {
    return DotProduct(v, v, length);
}

TTriWayDotProduct<float> TriWayDotProduct(const float* lhs, const float* rhs, ui32 length, unsigned mask) noexcept {
    TTriWayDotProduct<float> result;
    if (mask & static_cast<unsigned>(ETriWayDotProductComputeMask::LL)) {
        result.LL = L2NormSquared(lhs, length);
    }
    if (mask & static_cast<unsigned>(ETriWayDotProductComputeMask::LR)) {
        result.LR = DotProduct(lhs, rhs, length);
    }
    if (mask & static_cast<unsigned>(ETriWayDotProductComputeMask::RR)) {
        result.RR = L2NormSquared(rhs, length);
    }
    return result;
}

#endif // ARCADIA_SSE

i32 DotProductSlow(const i8* lhs, const i8* rhs, ui32 length) noexcept {
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
static Res DotProductSlowImpl(const Number* lhs, const Number* rhs, ui32 length) noexcept {
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

    while (length--) {
        s0 += static_cast<Res>(*lhs++) * static_cast<Res>(*rhs++);
    }

    return s0 + s1 + s2 + s3;
}

ui32 DotProductSlow(const ui8* lhs, const ui8* rhs, ui32 length) noexcept {
    return DotProductSlowImpl<ui32, ui8>(lhs, rhs, length);
}

i64 DotProductSlow(const i32* lhs, const i32* rhs, ui32 length) noexcept {
    return DotProductSlowImpl<i64, i32>(lhs, rhs, length);
}

float DotProductSlow(const float* lhs, const float* rhs, ui32 length) noexcept {
    return DotProductSlowImpl<float, float>(lhs, rhs, length);
}

double DotProductSlow(const double* lhs, const double* rhs, ui32 length) noexcept {
    return DotProductSlowImpl<double, double>(lhs, rhs, length);
}
