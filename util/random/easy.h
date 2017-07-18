#pragma once

#include "random.h"

namespace NPrivate {
    struct TRandom {
        inline operator unsigned char() {
            return RandomNumber<unsigned char>();
        }

        inline operator unsigned short() {
            return RandomNumber<unsigned short>();
        }

        inline operator unsigned int() {
            return RandomNumber<unsigned int>();
        }

        inline operator unsigned long() {
            return RandomNumber<unsigned long>();
        }

        inline operator unsigned long long() {
            return RandomNumber<unsigned long long>();
        }

        inline operator bool() {
            return RandomNumber<bool>();
        }

        inline operator float() {
            return RandomNumber<float>();
        }

        inline operator double() {
            return RandomNumber<double>();
        }

        inline operator long double() {
            return RandomNumber<long double>();
        }
    };
}

static inline ::NPrivate::TRandom Random() noexcept {
    return {};
}
