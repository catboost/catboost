#include "dot_product.h"
#include "dot_product_sse.h"
#include "dot_product_avx2.h"
#include "dot_product_simple.h"

#include <library/cpp/sse/sse.h>
#include <library/cpp/testing/common/env.h>
#include <util/system/compiler.h>
#include <util/generic/utility.h>
#include <util/system/cpu_id.h>
#include <util/system/env.h>

namespace NDotProductImpl {
    i32 (*DotProductI8Impl)(const i8* lhs, const i8* rhs, size_t length) noexcept = &DotProductSimple;
    ui32 (*DotProductUi8Impl)(const ui8* lhs, const ui8* rhs, size_t length) noexcept = &DotProductSimple;
    i64 (*DotProductI32Impl)(const i32* lhs, const i32* rhs, size_t length) noexcept = &DotProductSimple;
    float (*DotProductFloatImpl)(const float* lhs, const float* rhs, size_t length) noexcept = &DotProductSimple;
    double (*DotProductDoubleImpl)(const double* lhs, const double* rhs, size_t length) noexcept = &DotProductSimple;

    namespace {
        [[maybe_unused]] const int _ = [] {
            if (!FromYaTest() && GetEnv("Y_NO_AVX_IN_DOT_PRODUCT") == "" && NX86::HaveAVX2() && NX86::HaveFMA()) {
                DotProductI8Impl = &DotProductAvx2;
                DotProductUi8Impl = &DotProductAvx2;
                DotProductI32Impl = &DotProductAvx2;
                DotProductFloatImpl = &DotProductAvx2;
                DotProductDoubleImpl = &DotProductAvx2;
            } else {
#ifdef ARCADIA_SSE
                DotProductI8Impl = &DotProductSse;
                DotProductUi8Impl = &DotProductSse;
                DotProductI32Impl = &DotProductSse;
                DotProductFloatImpl = &DotProductSse;
                DotProductDoubleImpl = &DotProductSse;
#endif
            }
            return 0;
        }();
    }
}

#ifdef ARCADIA_SSE
float L2NormSquared(const float* v, size_t length) noexcept {
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
static TTriWayDotProduct<float> TriWayDotProductImpl(const float* lhs, const float* rhs, size_t length) noexcept {
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


TTriWayDotProduct<float> TriWayDotProduct(const float* lhs, const float* rhs, size_t length, unsigned mask) noexcept {
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

float L2NormSquared(const float* v, size_t length) noexcept {
    return DotProduct(v, v, length);
}

TTriWayDotProduct<float> TriWayDotProduct(const float* lhs, const float* rhs, size_t length, unsigned mask) noexcept {
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

namespace NDotProduct {
    void DisableAvx2() {
#ifdef ARCADIA_SSE
        NDotProductImpl::DotProductI8Impl = &DotProductSse;
        NDotProductImpl::DotProductUi8Impl = &DotProductSse;
        NDotProductImpl::DotProductI32Impl = &DotProductSse;
        NDotProductImpl::DotProductFloatImpl = &DotProductSse;
        NDotProductImpl::DotProductDoubleImpl = &DotProductSse;
#else
        NDotProductImpl::DotProductI8Impl = &DotProductSimple;
        NDotProductImpl::DotProductUi8Impl = &DotProductSimple;
        NDotProductImpl::DotProductI32Impl = &DotProductSimple;
        NDotProductImpl::DotProductFloatImpl = &DotProductSimple;
        NDotProductImpl::DotProductDoubleImpl = &DotProductSimple;
#endif
    }
}
