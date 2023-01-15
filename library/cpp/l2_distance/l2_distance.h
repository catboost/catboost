#pragma once

#include <util/system/types.h>
#include <util/generic/ymath.h>
#include <cmath>

namespace NPrivate {
    namespace NL2Distance {
        template <typename Number>
        inline Number L2DistanceSqrt(Number a) {
            return std::sqrt(a);
        }

        template <>
        inline ui64 L2DistanceSqrt(ui64 a) {
            // https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Binary_numeral_system_.28base_2.29
            ui64 res = 0;
            ui64 bit = static_cast<ui64>(1) << (sizeof(ui64) * 8 - 2);

            while (bit > a)
                bit >>= 2;

            while (bit != 0) {
                if (a >= res + bit) {
                    a -= (res + bit);
                    res = (res >> 1) + bit;
                } else {
                    res >>= 1;
                }
                bit >>= 2;
            }

            return res;
        }

        template <>
        inline ui32 L2DistanceSqrt(ui32 a) {
            return L2DistanceSqrt<ui64>(a);
        }

        // Special class to match argument type and result type.
        template <typename Arg>
        class TMatchArgumentResult {
        public:
            using TResult = Arg;
        };

        template <>
        class TMatchArgumentResult<i8> {
        public:
            using TResult = ui32;
        };

        template <>
        class TMatchArgumentResult<ui8> {
        public:
            using TResult = ui32;
        };

        template <>
        class TMatchArgumentResult<i32> {
        public:
            using TResult = ui64;
        };

        template <>
        class TMatchArgumentResult<ui32> {
        public:
            using TResult = ui64;
        };

    }

}

/**
 * sqr(l2_distance) = sum((a[i]-b[i])^2)
 * If target system does not support SSE2 Slow functions are used automatically.
 */
ui32 L2SqrDistance(const i8* a, const i8* b, int cnt);
ui32 L2SqrDistance(const ui8* a, const ui8* b, int cnt);
ui64 L2SqrDistance(const i32* a, const i32* b, int length);
ui64 L2SqrDistance(const ui32* a, const ui32* b, int length);
float L2SqrDistance(const float* a, const float* b, int length);
double L2SqrDistance(const double* a, const double* b, int length);
ui32 L2SqrDistanceUI4(const ui8* a, const ui8* b, int cnt);

ui32 L2SqrDistanceSlow(const i8* a, const i8* b, int cnt);
ui32 L2SqrDistanceSlow(const ui8* a, const ui8* b, int cnt);
ui64 L2SqrDistanceSlow(const i32* a, const i32* b, int length);
ui64 L2SqrDistanceSlow(const ui32* a, const ui32* b, int length);
float L2SqrDistanceSlow(const float* a, const float* b, int length);
double L2SqrDistanceSlow(const double* a, const double* b, int length);
ui32 L2SqrDistanceUI4Slow(const ui8* a, const ui8* b, int cnt);

/**
 * L2 distance = sqrt(sum((a[i]-b[i])^2))
 */
template <typename Number, typename Result = typename NPrivate::NL2Distance::TMatchArgumentResult<Number>::TResult>
inline Result L2Distance(const Number* a, const Number* b, int cnt) {
    return NPrivate::NL2Distance::L2DistanceSqrt(L2SqrDistance(a, b, cnt));
}

template <typename Number, typename Result = typename NPrivate::NL2Distance::TMatchArgumentResult<Number>::TResult>
inline Result L2DistanceSlow(const Number* a, const Number* b, int cnt) {
    return NPrivate::NL2Distance::L2DistanceSqrt(L2SqrDistanceSlow(a, b, cnt));
}

namespace NL2Distance {
    // You can use this structures as template function arguments.
    template <typename T>
    struct TL2Distance {
        using TResult = decltype(L2Distance(static_cast<const T*>(nullptr), static_cast<const T*>(nullptr), 0));
        inline TResult operator()(const T* a, const T* b, int length) const {
            return L2Distance(a, b, length);
        }
    };

    struct TL2DistanceUI4 {
        using TResult = ui32;
        inline TResult operator()(const ui8* a, const ui8* b, int lengtInBytes) const {
            return NPrivate::NL2Distance::L2DistanceSqrt(L2SqrDistanceUI4(a, b, lengtInBytes));
        }
    };

    template <typename T>
    struct TL2SqrDistance {
        using TResult = decltype(L2SqrDistance(static_cast<const T*>(nullptr), static_cast<const T*>(nullptr), 0));
        inline TResult operator()(const T* a, const T* b, int length) const {
            return L2SqrDistance(a, b, length);
        }
    };

    struct TL2SqrDistanceUI4 {
        using TResult = ui32;
        inline TResult operator()(const ui8* a, const ui8* b, int lengtInBytes) const {
            return L2SqrDistanceUI4(a, b, lengtInBytes);
        }
    };
}
