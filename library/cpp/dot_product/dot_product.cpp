#include "dot_product.h"
#include "dot_product_sse.h"
#include "dot_product_avx2.h"
#include "dot_product_vnni.h"
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
    float (*DotProductFloatI8Impl)(const float* lhs, const i8* rhs, size_t length) noexcept = &DotProductSimple;
    double (*DotProductDoubleImpl)(const double* lhs, const double* rhs, size_t length) noexcept = &DotProductSimple;

    TTriWayDotProduct<float> (*TriWayDotProductImpl)
        (const float* lhs, const float* rhs, size_t length, bool computeRR) noexcept = &TriWayDotProductSimple;

    TTriWayDotProductFloatI8 (*TriWayDotProductFloatI8Impl)
        (const float* lhs, const i8* rhs, size_t length) noexcept = &TriWayDotProductFloatI8Simple;

    TTriWayDotProduct<i32> (*TriWayDotProductI8Impl)
        (const i8* lhs, const i8* rhs, size_t length) noexcept = &TriWayDotProductI8Simple;

    namespace {
        template <size_t length>
        i32 DotProductFixedSimple(const i8* lhs, const i8* rhs) noexcept {
            return DotProductSimple(lhs, rhs, length);
        }

        template <size_t length>
        float DotProductFixedSimple(const float* lhs, const float* rhs) noexcept {
            return DotProductSimple(lhs, rhs, length);
        }

#ifdef ARCADIA_SSE
        template <size_t length>
        i32 DotProductFixedSse(const i8* lhs, const i8* rhs) noexcept {
            return DotProductSse(lhs, rhs, length);
        }

        template <size_t length>
        float DotProductFixedSse(const float* lhs, const float* rhs) noexcept {
            return DotProductSse(lhs, rhs, length);
        }
#endif
    }

    i32 (*DotProduct64I8Impl)(const i8* lhs, const i8* rhs) noexcept = &DotProductFixedSimple<64>;
    i32 (*DotProduct128I8Impl)(const i8* lhs, const i8* rhs) noexcept = &DotProductFixedSimple<128>;
    i32 (*DotProduct256I8Impl)(const i8* lhs, const i8* rhs) noexcept = &DotProductFixedSimple<256>;
    i32 (*DotProduct512I8Impl)(const i8* lhs, const i8* rhs) noexcept = &DotProductFixedSimple<512>;
    float (*DotProduct64FloatImpl)(const float* lhs, const float* rhs) noexcept = &DotProductFixedSimple<64>;
    float (*DotProduct128FloatImpl)(const float* lhs, const float* rhs) noexcept = &DotProductFixedSimple<128>;
    float (*DotProduct256FloatImpl)(const float* lhs, const float* rhs) noexcept = &DotProductFixedSimple<256>;
    float (*DotProduct512FloatImpl)(const float* lhs, const float* rhs) noexcept = &DotProductFixedSimple<512>;

    namespace {
        [[maybe_unused]] const int _ = [] {
            if (!FromYaTest() && GetEnv("Y_NO_AVX_IN_DOT_PRODUCT") == "" && NX86::HaveAVX2() && NX86::HaveFMA()) {
                DotProductUi8Impl = &DotProductAvx2;
                DotProductI32Impl = &DotProductAvx2;
                DotProductFloatImpl = &DotProductAvx2;
                DotProductFloatI8Impl = &DotProductFloatI8Avx2;
                DotProductDoubleImpl = &DotProductAvx2;
                TriWayDotProductImpl = &TriWayDotProductAvx2;
                TriWayDotProductFloatI8Impl = &TriWayDotProductFloatI8Avx2;
                TriWayDotProductI8Impl = &TriWayDotProductI8Avx2;
                DotProduct64FloatImpl = &DotProduct64Avx2;
                DotProduct128FloatImpl = &DotProduct128Avx2;
                DotProduct256FloatImpl = &DotProduct256Avx2;
                DotProduct512FloatImpl = &DotProduct512Avx2;

                if (GetEnv("Y_NO_VNNI_IN_DOT_PRODUCT") == "" && NX86::HaveAVX512VNNI() && NX86::HaveAVX512BW()) {
                    DotProductI8Impl = &DotProductVnni;
                    DotProduct64I8Impl = &DotProduct64Vnni;
                    DotProduct128I8Impl = &DotProduct128Vnni;
                    DotProduct256I8Impl = &DotProduct256Vnni;
                    DotProduct512I8Impl = &DotProduct512Vnni;
                } else {
                    DotProductI8Impl = &DotProductAvx2;
                    DotProduct64I8Impl = &DotProduct64Avx2;
                    DotProduct128I8Impl = &DotProduct128Avx2;
                    DotProduct256I8Impl = &DotProduct256Avx2;
                    DotProduct512I8Impl = &DotProduct512Avx2;
                }
            } else {
#ifdef ARCADIA_SSE
                DotProductI8Impl = &DotProductSse;
                DotProductUi8Impl = &DotProductSse;
                DotProductI32Impl = &DotProductSse;
                DotProductFloatImpl = &DotProductSse;
                DotProductFloatI8Impl = &DotProductSse;
                DotProductDoubleImpl = &DotProductSse;
                TriWayDotProductImpl = &TriWayDotProductSse;
                TriWayDotProductFloatI8Impl = &TriWayDotProductFloatI8Sse;
                TriWayDotProductI8Impl = &TriWayDotProductI8Sse;
                DotProduct64I8Impl = &DotProductFixedSse<64>;
                DotProduct128I8Impl = &DotProductFixedSse<128>;
                DotProduct256I8Impl = &DotProductFixedSse<256>;
                DotProduct512I8Impl = &DotProductFixedSse<512>;
                DotProduct64FloatImpl = &DotProductFixedSse<64>;
                DotProduct128FloatImpl = &DotProductFixedSse<128>;
                DotProduct256FloatImpl = &DotProductFixedSse<256>;
                DotProduct512FloatImpl = &DotProductFixedSse<512>;
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

TTriWayDotProduct<float> TriWayDotProduct(const float* lhs, const float* rhs, size_t length, unsigned mask) noexcept {
    mask &= 0b111;
    if (Y_LIKELY(mask == 0b111)) { // compute dot-product and length² of two vectors
        return NDotProductImpl::TriWayDotProductImpl(lhs, rhs, length, true);
    } else if (Y_LIKELY(mask == 0b110 || mask == 0b011)) { // compute dot-product and length² of one vector
        const bool computeLL = (mask == 0b110);
        if (!computeLL) {
            DoSwap(lhs, rhs);
        }
        auto result = NDotProductImpl::TriWayDotProductImpl(lhs, rhs, length, false);
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
        NDotProductImpl::DotProductFloatI8Impl = &DotProductSse;
        NDotProductImpl::DotProductDoubleImpl = &DotProductSse;
        NDotProductImpl::TriWayDotProductImpl = &TriWayDotProductSse;
        NDotProductImpl::TriWayDotProductFloatI8Impl = &TriWayDotProductFloatI8Sse;
        NDotProductImpl::TriWayDotProductI8Impl = &TriWayDotProductI8Sse;
        NDotProductImpl::DotProduct64I8Impl = &NDotProductImpl::DotProductFixedSse<64>;
        NDotProductImpl::DotProduct128I8Impl = &NDotProductImpl::DotProductFixedSse<128>;
        NDotProductImpl::DotProduct256I8Impl = &NDotProductImpl::DotProductFixedSse<256>;
        NDotProductImpl::DotProduct512I8Impl = &NDotProductImpl::DotProductFixedSse<512>;
        NDotProductImpl::DotProduct64FloatImpl = &NDotProductImpl::DotProductFixedSse<64>;
        NDotProductImpl::DotProduct128FloatImpl = &NDotProductImpl::DotProductFixedSse<128>;
        NDotProductImpl::DotProduct256FloatImpl = &NDotProductImpl::DotProductFixedSse<256>;
        NDotProductImpl::DotProduct512FloatImpl = &NDotProductImpl::DotProductFixedSse<512>;
#else
        NDotProductImpl::DotProductI8Impl = &DotProductSimple;
        NDotProductImpl::DotProductUi8Impl = &DotProductSimple;
        NDotProductImpl::DotProductI32Impl = &DotProductSimple;
        NDotProductImpl::DotProductFloatImpl = &DotProductSimple;
        NDotProductImpl::DotProductFloatI8Impl = &DotProductSimple;
        NDotProductImpl::DotProductDoubleImpl = &DotProductSimple;
        NDotProductImpl::TriWayDotProductImpl = &TriWayDotProductSimple;
        NDotProductImpl::TriWayDotProductFloatI8Impl = &TriWayDotProductFloatI8Simple;
        NDotProductImpl::TriWayDotProductI8Impl = &TriWayDotProductI8Simple;
        NDotProductImpl::DotProduct64I8Impl = &NDotProductImpl::DotProductFixedSimple<64>;
        NDotProductImpl::DotProduct128I8Impl = &NDotProductImpl::DotProductFixedSimple<128>;
        NDotProductImpl::DotProduct256I8Impl = &NDotProductImpl::DotProductFixedSimple<256>;
        NDotProductImpl::DotProduct512I8Impl = &NDotProductImpl::DotProductFixedSimple<512>;
        NDotProductImpl::DotProduct64FloatImpl = &NDotProductImpl::DotProductFixedSimple<64>;
        NDotProductImpl::DotProduct128FloatImpl = &NDotProductImpl::DotProductFixedSimple<128>;
        NDotProductImpl::DotProduct256FloatImpl = &NDotProductImpl::DotProductFixedSimple<256>;
        NDotProductImpl::DotProduct512FloatImpl = &NDotProductImpl::DotProductFixedSimple<512>;
#endif
    }
}
