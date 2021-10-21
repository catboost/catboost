#include "dot_product_simple.h"

namespace {
    template <typename Res, typename Number>
    static Res DotProductSimpleImpl(const Number* lhs, const Number* rhs, size_t length) noexcept {
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
}

float DotProductSimple(const float* lhs, const float* rhs, size_t length) noexcept {
    return DotProductSimpleImpl<float, float>(lhs, rhs, length);
}

double DotProductSimple(const double* lhs, const double* rhs, size_t length) noexcept {
    return DotProductSimpleImpl<double, double>(lhs, rhs, length);
}

ui32 DotProductUI4Simple(const ui8* lhs, const ui8* rhs, size_t lengtInBytes) noexcept {
    ui32 res = 0;
    for (size_t i = 0; i < lengtInBytes; ++i) {
        res += static_cast<ui32>(lhs[i] & 0x0f) * static_cast<ui32>(rhs[i] & 0x0f);
        res += static_cast<ui32>(lhs[i] & 0xf0) * static_cast<ui32>(rhs[i] & 0xf0) >> 8;
    }
    return res;
}
