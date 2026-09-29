#pragma once

#include "common.h"

#include <util/system/types.h>
#include <util/system/compiler.h>

#include <numeric>

/**
 * Dot product (Inner product or scalar product) implementation using SIMD when possible.
 */
namespace NDotProductImpl {
    extern i32 (*DotProductI8Impl)(const i8* lhs, const i8* rhs, size_t length) noexcept;
    extern ui32 (*DotProductUi8Impl)(const ui8* lhs, const ui8* rhs, size_t length) noexcept;
    extern i64 (*DotProductI32Impl)(const i32* lhs, const i32* rhs, size_t length) noexcept;
    extern float (*DotProductFloatImpl)(const float* lhs, const float* rhs, size_t length) noexcept;
    extern float (*DotProductFloatI8Impl)(const float* lhs, const i8* rhs, size_t length) noexcept;
    extern double (*DotProductDoubleImpl)(const double* lhs, const double* rhs, size_t length) noexcept;

    extern TTriWayDotProduct<float> (*TriWayDotProductImpl)
        (const float* lhs, const float* rhs, size_t length, bool computeRR) noexcept;

    extern TTriWayDotProductFloatI8 (*TriWayDotProductFloatI8Impl)
        (const float* lhs, const i8* rhs, size_t length) noexcept;

    extern TTriWayDotProduct<i32> (*TriWayDotProductI8Impl)
        (const i8* lhs, const i8* rhs, size_t length) noexcept;

    extern i32 (*DotProduct64I8Impl)(const i8* lhs, const i8* rhs) noexcept;
    extern float (*DotProduct64FloatImpl)(const float* lhs, const float* rhs) noexcept;
    extern i32 (*DotProduct128I8Impl)(const i8* lhs, const i8* rhs) noexcept;
    extern float (*DotProduct128FloatImpl)(const float* lhs, const float* rhs) noexcept;
    extern i32 (*DotProduct256I8Impl)(const i8* lhs, const i8* rhs) noexcept;
    extern float (*DotProduct256FloatImpl)(const float* lhs, const float* rhs) noexcept;
    extern i32 (*DotProduct512I8Impl)(const i8* lhs, const i8* rhs) noexcept;
    extern float (*DotProduct512FloatImpl)(const float* lhs, const float* rhs) noexcept;
}

Y_PURE_FUNCTION
inline i32 DotProduct(const i8* lhs, const i8* rhs, size_t length) noexcept {
    return NDotProductImpl::DotProductI8Impl(lhs, rhs, length);
}

Y_PURE_FUNCTION
inline ui32 DotProduct(const ui8* lhs, const ui8* rhs, size_t length) noexcept {
    return NDotProductImpl::DotProductUi8Impl(lhs, rhs, length);
}

Y_PURE_FUNCTION
inline i64 DotProduct(const i32* lhs, const i32* rhs, size_t length) noexcept {
    return NDotProductImpl::DotProductI32Impl(lhs, rhs, length);
}

Y_PURE_FUNCTION
inline float DotProduct(const float* lhs, const float* rhs, size_t length) noexcept {
    return NDotProductImpl::DotProductFloatImpl(lhs, rhs, length);
}

Y_PURE_FUNCTION
inline float DotProduct(const float* lhs, const i8* rhs, size_t length) noexcept {
    return NDotProductImpl::DotProductFloatI8Impl(lhs, rhs, length);
}

Y_PURE_FUNCTION
inline double DotProduct(const double* lhs, const double* rhs, size_t length) noexcept {
    return NDotProductImpl::DotProductDoubleImpl(lhs, rhs, length);
}

/**
 * Dot product of vectors with fixed length 64, 128, 256 or 512 (`DotProduct256(l, r) == DotProduct(l, r, 256)`).
 * Fully unrolled, without tail handling and length dispatch, uses AVX512 VNNI for i8 when available.
 * Float result may differ from `DotProduct` in the last bits because of different summation order.
 */
Y_PURE_FUNCTION
inline i32 DotProduct64(const i8* lhs, const i8* rhs) noexcept {
    return NDotProductImpl::DotProduct64I8Impl(lhs, rhs);
}

Y_PURE_FUNCTION
inline float DotProduct64(const float* lhs, const float* rhs) noexcept {
    return NDotProductImpl::DotProduct64FloatImpl(lhs, rhs);
}

Y_PURE_FUNCTION
inline i32 DotProduct128(const i8* lhs, const i8* rhs) noexcept {
    return NDotProductImpl::DotProduct128I8Impl(lhs, rhs);
}

Y_PURE_FUNCTION
inline float DotProduct128(const float* lhs, const float* rhs) noexcept {
    return NDotProductImpl::DotProduct128FloatImpl(lhs, rhs);
}

Y_PURE_FUNCTION
inline i32 DotProduct256(const i8* lhs, const i8* rhs) noexcept {
    return NDotProductImpl::DotProduct256I8Impl(lhs, rhs);
}

Y_PURE_FUNCTION
inline float DotProduct256(const float* lhs, const float* rhs) noexcept {
    return NDotProductImpl::DotProduct256FloatImpl(lhs, rhs);
}

Y_PURE_FUNCTION
inline i32 DotProduct512(const i8* lhs, const i8* rhs) noexcept {
    return NDotProductImpl::DotProduct512I8Impl(lhs, rhs);
}

Y_PURE_FUNCTION
inline float DotProduct512(const float* lhs, const float* rhs) noexcept {
    return NDotProductImpl::DotProduct512FloatImpl(lhs, rhs);
}

/**
 * Dot product to itself
 */
Y_PURE_FUNCTION
float L2NormSquared(const float* v, size_t length) noexcept;

// TODO(yazevnul): make `L2NormSquared` for double, this should be faster than `DotProduct`
// where `lhs == rhs` because it will save N load instructions.

Y_PURE_FUNCTION
TTriWayDotProduct<float> TriWayDotProduct(const float* lhs, const float* rhs, size_t length, unsigned mask) noexcept;

/**
 * For two vectors L and R computes 3 dot-products: L·L, L·R, R·R
 */
Y_PURE_FUNCTION
static inline TTriWayDotProduct<float> TriWayDotProduct(
    const float* lhs,
    const float* rhs,
    size_t length,
    ETriWayDotProductComputeMask mask = ETriWayDotProductComputeMask::All) noexcept
{
    return TriWayDotProduct(lhs, rhs, length, static_cast<unsigned>(mask));
}

Y_PURE_FUNCTION
inline TTriWayDotProductFloatI8 TriWayDotProduct(
    const float* lhs,
    const i8* rhs,
    size_t length) noexcept
{
    return NDotProductImpl::TriWayDotProductFloatI8Impl(lhs, rhs, length);
}

Y_PURE_FUNCTION
inline TTriWayDotProduct<i32> TriWayDotProduct(
    const i8* lhs,
    const i8* rhs,
    size_t length) noexcept
{
    return NDotProductImpl::TriWayDotProductI8Impl(lhs, rhs, length);
}

namespace NDotProduct {
    // Simpler wrapper allowing to use this functions as template argument.
    template <typename T>
    struct TDotProduct {
        using TResult = decltype(DotProduct(static_cast<const T*>(nullptr), static_cast<const T*>(nullptr), 0));
        Y_PURE_FUNCTION
        inline TResult operator()(const T* l, const T* r, size_t length) const {
            return DotProduct(l, r, length);
        }
    };

    void DisableAvx2();
}

