#pragma once

#include <library/cpp/sse/sse.h>

#include <util/system/types.h>
#include <util/generic/ymath.h>
#include <util/system/align.h>
#include <util/system/platform.h>

namespace NL1Distance {
    namespace NPrivate {
        template <typename T>
        inline T AbsDelta(T a, T b) {
            if (a < b)
                return b - a;
            return a - b;
        }

        template <typename Result, typename Number>
        inline Result L1DistanceImpl(const Number* lhs, const Number* rhs, int length) {
            Result sum = 0;

            for (int i = 0; i < length; i++)
                sum += AbsDelta(lhs[i], rhs[i]);

            return sum;
        }

        template <typename Result, typename Number>
        inline Result L1DistanceImpl2(const Number* lhs, const Number* rhs, int length) {
            Result s0 = 0;
            Result s1 = 0;

            while (length >= 2) {
                s0 += AbsDelta(lhs[0], rhs[0]);
                s1 += AbsDelta(lhs[1], rhs[1]);
                lhs += 2;
                rhs += 2;
                length -= 2;
            }

            while (length--)
                s0 += AbsDelta(*lhs++, *rhs++);

            return s0 + s1;
        }

        template <typename Result, typename Number>
        inline Result L1DistanceImpl4(const Number* lhs, const Number* rhs, int length) {
            Result s0 = 0;
            Result s1 = 0;
            Result s2 = 0;
            Result s3 = 0;

            while (length >= 4) {
                s0 += AbsDelta(lhs[0], rhs[0]);
                s1 += AbsDelta(lhs[1], rhs[1]);
                s2 += AbsDelta(lhs[2], rhs[2]);
                s3 += AbsDelta(lhs[3], rhs[3]);
                lhs += 4;
                rhs += 4;
                length -= 4;
            }

            while (length--)
                s0 += AbsDelta(*lhs++, *rhs++);

            return s0 + s1 + s2 + s3;
        }

        template <typename Result>
        inline Result L1DistanceImplUI4(const ui8* lhs, const ui8* rhs, int lengtInBytes) {
            Result sum = 0;

            for (int i = 0; i < lengtInBytes; ++i) {
                sum += AbsDelta(lhs[i] & 0x0f, rhs[i] & 0x0f);
                sum += AbsDelta(lhs[i] & 0xf0, rhs[i] & 0xf0) >> 4;
            }

            return sum;
        }

#ifdef ARCADIA_SSE
        static const __m128i MASK_UI4_1 = _mm_set_epi8(0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f);
        static const __m128i MASK_UI4_2 = _mm_set_epi8(0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0);


        Y_FORCE_INLINE ui32 L1Distance96Ui8(const ui8* lhs, const ui8* rhs) {
            __m128i x1 = _mm_loadu_si128((const __m128i*)&lhs[0]);
            __m128i y1 = _mm_loadu_si128((const __m128i*)&rhs[0]);

            __m128i sum = _mm_sad_epu8(x1, y1);

            __m128i x2 = _mm_loadu_si128((const __m128i*)&lhs[16]);
            __m128i y2 = _mm_loadu_si128((const __m128i*)&rhs[16]);

            sum = _mm_add_epi64(sum, _mm_sad_epu8(x2, y2));

            __m128i x3 = _mm_loadu_si128((const __m128i*)&lhs[32]);
            __m128i y3 = _mm_loadu_si128((const __m128i*)&rhs[32]);

            sum = _mm_add_epi64(sum, _mm_sad_epu8(x3, y3));

            __m128i x4 = _mm_loadu_si128((const __m128i*)&lhs[48]);
            __m128i y4 = _mm_loadu_si128((const __m128i*)&rhs[48]);

            sum = _mm_add_epi64(sum, _mm_sad_epu8(x4, y4));

            __m128i x5 = _mm_loadu_si128((const __m128i*)&lhs[64]);
            __m128i y5 = _mm_loadu_si128((const __m128i*)&rhs[64]);

            sum = _mm_add_epi64(sum, _mm_sad_epu8(x5, y5));

            __m128i x6 = _mm_loadu_si128((const __m128i*)&lhs[80]);
            __m128i y6 = _mm_loadu_si128((const __m128i*)&rhs[80]);

            sum = _mm_add_epi64(sum, _mm_sad_epu8(x6, y6));
            return _mm_cvtsi128_si32(sum) + _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 2, 2, 2)));
        }

        Y_FORCE_INLINE ui32 L1Distance96Ui4(const ui8* lhs, const ui8* rhs) {
            __m128i x1 = _mm_loadu_si128((const __m128i*)&lhs[0]);
            __m128i y1 = _mm_loadu_si128((const __m128i*)&rhs[0]);
            __m128i sum1 = _mm_sad_epu8(_mm_and_si128(x1, MASK_UI4_1), _mm_and_si128(y1, MASK_UI4_1));
            __m128i sum2 = _mm_sad_epu8(_mm_and_si128(x1, MASK_UI4_2), _mm_and_si128(y1, MASK_UI4_2));

            __m128i x2 = _mm_loadu_si128((const __m128i*)&lhs[16]);
            __m128i y2 = _mm_loadu_si128((const __m128i*)&rhs[16]);
            sum1 = _mm_add_epi64(sum1, _mm_sad_epu8(_mm_and_si128(x2, MASK_UI4_1), _mm_and_si128(y2, MASK_UI4_1)));
            sum2 = _mm_add_epi64(sum2, _mm_sad_epu8(_mm_and_si128(x2, MASK_UI4_2), _mm_and_si128(y2, MASK_UI4_2)));

            __m128i x3 = _mm_loadu_si128((const __m128i*)&lhs[32]);
            __m128i y3 = _mm_loadu_si128((const __m128i*)&rhs[32]);
            sum1 = _mm_add_epi64(sum1, _mm_sad_epu8(_mm_and_si128(x3, MASK_UI4_1), _mm_and_si128(y3, MASK_UI4_1)));
            sum2 = _mm_add_epi64(sum2, _mm_sad_epu8(_mm_and_si128(x3, MASK_UI4_2), _mm_and_si128(y3, MASK_UI4_2)));

            __m128i x4 = _mm_loadu_si128((const __m128i*)&lhs[48]);
            __m128i y4 = _mm_loadu_si128((const __m128i*)&rhs[48]);
            sum1 = _mm_add_epi64(sum1, _mm_sad_epu8(_mm_and_si128(x4, MASK_UI4_1), _mm_and_si128(y4, MASK_UI4_1)));
            sum2 = _mm_add_epi64(sum2, _mm_sad_epu8(_mm_and_si128(x4, MASK_UI4_2), _mm_and_si128(y4, MASK_UI4_2)));

            __m128i x5 = _mm_loadu_si128((const __m128i*)&lhs[64]);
            __m128i y5 = _mm_loadu_si128((const __m128i*)&rhs[64]);
            sum1 = _mm_add_epi64(sum1, _mm_sad_epu8(_mm_and_si128(x5, MASK_UI4_1), _mm_and_si128(y5, MASK_UI4_1)));
            sum2 = _mm_add_epi64(sum2, _mm_sad_epu8(_mm_and_si128(x5, MASK_UI4_2), _mm_and_si128(y5, MASK_UI4_2)));

            __m128i x6 = _mm_loadu_si128((const __m128i*)&lhs[80]);
            __m128i y6 = _mm_loadu_si128((const __m128i*)&rhs[80]);
            sum1 = _mm_add_epi64(sum1, _mm_sad_epu8(_mm_and_si128(x6, MASK_UI4_1), _mm_and_si128(y6, MASK_UI4_1)));
            sum2 = _mm_add_epi64(sum2, _mm_sad_epu8(_mm_and_si128(x6, MASK_UI4_2), _mm_and_si128(y6, MASK_UI4_2)));

            return _mm_cvtsi128_si32(sum1) + _mm_cvtsi128_si32(_mm_shuffle_epi32(sum1, _MM_SHUFFLE(2, 2, 2, 2))) +
                   ((_mm_cvtsi128_si32(sum2) + _mm_cvtsi128_si32(_mm_shuffle_epi32(sum2, _MM_SHUFFLE(2, 2, 2, 2)))) >> 4);
        }
#endif // ARCADIA_SSE
    }  // namespace NPrivate
}

/**
 * L1Distance (sum(abs(l[i] - r[i]))) implementation using SSE when possible.
 */
#ifdef ARCADIA_SSE

Y_FORCE_INLINE ui32 L1Distance(const i8* lhs, const i8* rhs, int length) {
    static const __m128i unsignedToSignedDiff = _mm_set_epi8(
        -128, -128, -128, -128, -128, -128, -128, -128,
        -128, -128, -128, -128, -128, -128, -128, -128);
    __m128i resVec = _mm_setzero_si128();

    while (length >= 16) {
        __m128i lVec = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)lhs), unsignedToSignedDiff);
        __m128i rVec = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)rhs), unsignedToSignedDiff);

        resVec = _mm_add_epi64(_mm_sad_epu8(lVec, rVec), resVec);

        lhs += 16;
        rhs += 16;
        length -= 16;
    }

    alignas(16) i64 res[2];
    _mm_store_si128((__m128i*)res, resVec);
    ui32 sum = res[0] + res[1];
    for (int i = 0; i < length; ++i) {
        const i32 diff = static_cast<i32>(lhs[i]) - static_cast<i32>(rhs[i]);
        sum += (diff >= 0) ? diff : -diff;
    }

    return sum;
}

Y_FORCE_INLINE ui32 L1Distance(const ui8* lhs, const ui8* rhs, int length) {
    if (length == 96)
        return NL1Distance::NPrivate::L1Distance96Ui8(lhs, rhs);

    int l16 = length & (~15);
    __m128i sum = _mm_setzero_si128();

    if ((reinterpret_cast<uintptr_t>(lhs) & 0x0f) || (reinterpret_cast<uintptr_t>(rhs) & 0x0f)) {
        for (int i = 0; i < l16; i += 16) {
            __m128i a = _mm_loadu_si128((const __m128i*)(&lhs[i]));
            __m128i b = _mm_loadu_si128((const __m128i*)(&rhs[i]));

            sum = _mm_add_epi64(sum, _mm_sad_epu8(a, b));
        }
    } else {
        for (int i = 0; i < l16; i += 16) {
            __m128i sum_ab = _mm_sad_epu8(*(const __m128i*)(&lhs[i]), *(const __m128i*)(&rhs[i]));
            sum = _mm_add_epi64(sum, sum_ab);
        }
    }

    if (l16 == length)
        return _mm_cvtsi128_si32(sum) + _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 2, 2, 2)));

    int l4 = length & (~3);
    for (int i = l16; i < l4; i += 4) {
        __m128i a = _mm_set_epi32(*((const ui32*)&lhs[i]), 0, 0, 0);
        __m128i b = _mm_set_epi32(*((const ui32*)&rhs[i]), 0, 0, 0);
        sum = _mm_add_epi64(sum, _mm_sad_epu8(a, b));
    }

    ui32 res = _mm_cvtsi128_si32(sum) + _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 2, 2, 2)));

    for (int i = l4; i < length; i++)
        res += lhs[i] < rhs[i] ? rhs[i] - lhs[i] : lhs[i] - rhs[i];

    return res;
}

Y_FORCE_INLINE ui32 L1DistanceUI4(const ui8* lhs, const ui8* rhs, int lengtInBytes) {

    if (lengtInBytes == 96)
        return NL1Distance::NPrivate::L1Distance96Ui4(lhs, rhs);

    int l16 = lengtInBytes & (~15);
    __m128i sum1 = _mm_setzero_si128();
    __m128i sum2 = _mm_setzero_si128();

    for (int i = 0; i < l16; i += 16) {
        __m128i a = _mm_loadu_si128((const __m128i*)(&lhs[i]));
        __m128i b = _mm_loadu_si128((const __m128i*)(&rhs[i]));

        sum1 = _mm_add_epi64(sum1, _mm_sad_epu8(_mm_and_si128(a, NL1Distance::NPrivate::MASK_UI4_1), _mm_and_si128(b, NL1Distance::NPrivate::MASK_UI4_1)));
        sum2 = _mm_add_epi64(sum2, _mm_sad_epu8(_mm_and_si128(a, NL1Distance::NPrivate::MASK_UI4_2), _mm_and_si128(b, NL1Distance::NPrivate::MASK_UI4_2)));
    }

    if (l16 == lengtInBytes)
        return _mm_cvtsi128_si32(sum1) + _mm_cvtsi128_si32(_mm_shuffle_epi32(sum1, _MM_SHUFFLE(2, 2, 2, 2))) +
                ((_mm_cvtsi128_si32(sum2) + _mm_cvtsi128_si32(_mm_shuffle_epi32(sum2, _MM_SHUFFLE(2, 2, 2, 2)))) >> 4);

    int l4 = lengtInBytes & (~3);
    for (int i = l16; i < l4; i += 4) {
        __m128i a = _mm_set_epi32(*((const ui32*)&lhs[i]), 0, 0, 0);
        __m128i b = _mm_set_epi32(*((const ui32*)&rhs[i]), 0, 0, 0);
        sum1 = _mm_add_epi64(sum1, _mm_sad_epu8(_mm_and_si128(a, NL1Distance::NPrivate::MASK_UI4_1), _mm_and_si128(b, NL1Distance::NPrivate::MASK_UI4_1)));
        sum2 = _mm_add_epi64(sum2, _mm_sad_epu8(_mm_and_si128(a, NL1Distance::NPrivate::MASK_UI4_2), _mm_and_si128(b, NL1Distance::NPrivate::MASK_UI4_2)));
    }

    ui32 res = _mm_cvtsi128_si32(sum1) + _mm_cvtsi128_si32(_mm_shuffle_epi32(sum1, _MM_SHUFFLE(2, 2, 2, 2))) +
                ((_mm_cvtsi128_si32(sum2) + _mm_cvtsi128_si32(_mm_shuffle_epi32(sum2, _MM_SHUFFLE(2, 2, 2, 2)))) >> 4);

    for (int i = l4; i < lengtInBytes; ++i) {
        ui8 a1 = lhs[i] & 0x0f;
        ui8 a2 = (lhs[i] & 0xf0) >> 4;
        ui8 b1 = rhs[i] & 0x0f;
        ui8 b2 = (rhs[i] & 0xf0) >> 4;
        res += a1 < b1 ? b1 - a1 : a1 - b1;
        res += a2 < b2 ? b2 - a2 : a2 - b2;
    }

    return res;
}

Y_FORCE_INLINE ui64 L1Distance(const i32* lhs, const i32* rhs, int length) {
    __m128i zero = _mm_setzero_si128();
    __m128i res = zero;

    while (length >= 4) {
        __m128i a = _mm_loadu_si128((const __m128i*)lhs);
        __m128i b = _mm_loadu_si128((const __m128i*)rhs);
        __m128i mask = _mm_cmpgt_epi32(a, b);
        __m128i a2 = _mm_and_si128(mask, _mm_sub_epi32(a, b));
        b = _mm_andnot_si128(mask, _mm_sub_epi32(b, a));
        a = _mm_or_si128(a2, b);
        res = _mm_add_epi64(_mm_unpackhi_epi32(a, zero), res);
        res = _mm_add_epi64(_mm_unpacklo_epi32(a, zero), res);
        rhs += 4;
        lhs += 4;
        length -= 4;
    }

    alignas(16) ui64 r[2];
    _mm_store_si128((__m128i*)r, res);
    ui64 sum = r[0] + r[1];

    while (length) {
        sum += lhs[0] < rhs[0] ? rhs[0] - lhs[0] : lhs[0] - rhs[0];
        ++lhs;
        ++rhs;
        --length;
    }

    return sum;
}

Y_FORCE_INLINE ui64 L1Distance(const ui32* lhs, const ui32* rhs, int length) {
    __m128i zero = _mm_setzero_si128();
    __m128i shift = _mm_set1_epi32(0x80000000);
    __m128i res = zero;

    while (length >= 4) {
        __m128i a = _mm_add_epi32(_mm_loadu_si128((const __m128i*)lhs), shift);
        __m128i b = _mm_add_epi32(_mm_loadu_si128((const __m128i*)rhs), shift);
        __m128i mask = _mm_cmpgt_epi32(a, b);
        __m128i a2 = _mm_and_si128(mask, _mm_sub_epi32(a, b));
        b = _mm_andnot_si128(mask, _mm_sub_epi32(b, a));
        a = _mm_or_si128(a2, b);
        res = _mm_add_epi64(_mm_unpackhi_epi32(a, zero), res);
        res = _mm_add_epi64(_mm_unpacklo_epi32(a, zero), res);
        rhs += 4;
        lhs += 4;
        length -= 4;
    }

    alignas(16) ui64 r[2];
    _mm_store_si128((__m128i*)r, res);
    ui64 sum = r[0] + r[1];

    while (length) {
        sum += lhs[0] < rhs[0] ? rhs[0] - lhs[0] : lhs[0] - rhs[0];
        ++lhs;
        ++rhs;
        --length;
    }

    return sum;
}

Y_FORCE_INLINE float L1Distance(const float* lhs, const float* rhs, int length) {
    __m128 res = _mm_setzero_ps();
    __m128 absMask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));

    while (length >= 4) {
        __m128 a = _mm_loadu_ps(lhs);
        __m128 b = _mm_loadu_ps(rhs);
        __m128 d = _mm_sub_ps(a, b);
        res = _mm_add_ps(_mm_and_ps(d, absMask), res);
        rhs += 4;
        lhs += 4;
        length -= 4;
    }

    alignas(16) float r[4];
    _mm_store_ps(r, res);
    float sum = r[0] + r[1] + r[2] + r[3];

    while (length) {
        sum += std::abs(*lhs - *rhs);
        ++lhs;
        ++rhs;
        --length;
    }

    return sum;
}

Y_FORCE_INLINE double L1Distance(const double* lhs, const double* rhs, int length) {
    __m128d res = _mm_setzero_pd();
    __m128d absMask = _mm_castsi128_pd(_mm_set_epi32(0x7fffffff, 0xffffffff, 0x7fffffff, 0xffffffff));

    while (length >= 2) {
        __m128d a = _mm_loadu_pd(lhs);
        __m128d b = _mm_loadu_pd(rhs);
        __m128d d = _mm_sub_pd(a, b);
        res = _mm_add_pd(_mm_and_pd(d, absMask), res);
        rhs += 2;
        lhs += 2;
        length -= 2;
    }

    alignas(16) double r[2];
    _mm_store_pd(r, res);
    double sum = r[0] + r[1];

    while (length) {
        sum += std::abs(*lhs - *rhs);
        ++lhs;
        ++rhs;
        --length;
    }

    return sum;
}

#else // ARCADIA_SSE

inline ui32 L1Distance(const i8* lhs, const i8* rhs, int length) {
    return NL1Distance::NPrivate::L1DistanceImpl<ui32, i8>(lhs, rhs, length);
}

inline ui32 L1Distance(const ui8* lhs, const ui8* rhs, int length) {
    return NL1Distance::NPrivate::L1DistanceImpl<ui32, ui8>(lhs, rhs, length);
}

inline ui32 L1DistanceUI4(const ui8* lhs, const ui8* rhs, int lengtInBytes) {
    return NL1Distance::NPrivate::L1DistanceImplUI4<ui32>(lhs, rhs, lengtInBytes);
}

inline ui64 L1Distance(const ui32* lhs, const ui32* rhs, int length) {
    return NL1Distance::NPrivate::L1DistanceImpl2<ui64, ui32>(lhs, rhs, length);
}

inline ui64 L1Distance(const i32* lhs, const i32* rhs, int length) {
    return NL1Distance::NPrivate::L1DistanceImpl2<ui64, i32>(lhs, rhs, length);
}

inline float L1Distance(const float* lhs, const float* rhs, int length) {
    return NL1Distance::NPrivate::L1DistanceImpl4<float, float>(lhs, rhs, length);
}

inline double L1Distance(const double* lhs, const double* rhs, int length) {
    return NL1Distance::NPrivate::L1DistanceImpl4<double, double>(lhs, rhs, length);
}

#endif // _sse_

/**
 * L1Distance (sum(abs(l[i] - r[i]))) implementation without SSE.
 */
inline ui32 L1DistanceSlow(const i8* lhs, const i8* rhs, int length) {
    return NL1Distance::NPrivate::L1DistanceImpl<ui32, i8>(lhs, rhs, length);
}

inline ui32 L1DistanceSlow(const ui8* lhs, const ui8* rhs, int length) {
    return NL1Distance::NPrivate::L1DistanceImpl<ui32, ui8>(lhs, rhs, length);
}

inline ui32 L1DistanceUI4Slow(const ui8* lhs, const ui8* rhs, int lengtInBytes) {
    return NL1Distance::NPrivate::L1DistanceImplUI4<ui32>(lhs, rhs, lengtInBytes);
}

inline ui64 L1DistanceSlow(const ui32* lhs, const ui32* rhs, int length) {
    return NL1Distance::NPrivate::L1DistanceImpl2<ui64, ui32>(lhs, rhs, length);
}

inline ui64 L1DistanceSlow(const i32* lhs, const i32* rhs, int length) {
    return NL1Distance::NPrivate::L1DistanceImpl2<ui64, i32>(lhs, rhs, length);
}

inline float L1DistanceSlow(const float* lhs, const float* rhs, int length) {
    return NL1Distance::NPrivate::L1DistanceImpl4<float, float>(lhs, rhs, length);
}

inline double L1DistanceSlow(const double* lhs, const double* rhs, int length) {
    return NL1Distance::NPrivate::L1DistanceImpl4<double, double>(lhs, rhs, length);
}

namespace NL1Distance {
    // Simpler wrapper allowing to use this functions as template argument.
    template <typename T>
    struct TL1Distance {
        using TResult = decltype(L1Distance(static_cast<const T*>(nullptr), static_cast<const T*>(nullptr), 0));

        inline TResult operator()(const T* a, const T* b, int length) const {
            return L1Distance(a, b, length);
        }
    };

    struct TL1DistanceUI4 {
        using TResult = ui32;

        inline TResult operator()(const ui8* a, const ui8* b, int lengtInBytes) const {
            return L1DistanceUI4(a, b, lengtInBytes);
        }
    };
}
