#include "dot_product_vnni.h"
#include "dot_product_avx2.h"

#include <util/system/compiler.h>
#include <util/system/platform.h>

#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512F__)

#include <immintrin.h>

namespace {
    Y_FORCE_INLINE __m512i Load512i(const void* ptr) {
        return _mm512_loadu_si512(ptr);
    }

    Y_FORCE_INLINE __m512i CorrectSignedLhsDotProduct(
        const __m512i biasedDotProduct,
        const __m512i rhsSum) noexcept
    {
        return _mm512_sub_epi32(biasedDotProduct, _mm512_slli_epi32(rhsSum, 7));
    }

    Y_FORCE_INLINE void AccumulateDotProductI8Vnni(
        __m512i& dotProduct,
        const i8* lhs,
        const i8* rhs,
        const __m512i zero,
        const __m512i signBit,
        const __m512i ones) noexcept
    {
        const __m512i l = Load512i(lhs);
        const __m512i r = Load512i(rhs);
        const __m512i unsignedL = _mm512_xor_si512(l, signBit);

        // AVX512 VNNI has u8*i8 dot product. For signed lhs we compute
        // (lhs + 128) * rhs and immediately subtract 128 * sum(rhs).
        // Correcting each 64-byte chunk keeps the accumulator in the same
        // range as a regular signed i8 dot product accumulator instead of
        // accumulating a much larger biased intermediate value.
        const __m512i biasedDotProduct = _mm512_dpbusd_epi32(zero, unsignedL, r);
        const __m512i rhsSum = _mm512_dpbusd_epi32(zero, ones, r);
        dotProduct = _mm512_add_epi32(dotProduct, CorrectSignedLhsDotProduct(biasedDotProduct, rhsSum));
    }

    Y_FORCE_INLINE void AccumulateTailDotProductI8Vnni(
        __m512i& dotProduct,
        const i8* lhs,
        const i8* rhs,
        const size_t length,
        const __m512i zero,
        const __m512i signBit,
        const __m512i ones) noexcept
    {
        const __mmask64 mask = (ui64(1) << length) - 1;
        const __m512i l = _mm512_maskz_loadu_epi8(mask, lhs);
        const __m512i r = _mm512_maskz_loadu_epi8(mask, rhs);
        const __m512i unsignedL = _mm512_xor_si512(l, signBit);

        const __m512i biasedDotProduct = _mm512_dpbusd_epi32(zero, unsignedL, r);
        const __m512i rhsSum = _mm512_dpbusd_epi32(zero, ones, r);
        dotProduct = _mm512_add_epi32(dotProduct, CorrectSignedLhsDotProduct(biasedDotProduct, rhsSum));
    }

    i32 HsumI32(const __m512i v) noexcept {
        __m128i sum = _mm512_castsi512_si128(v);
        sum = _mm_add_epi32(sum, _mm512_extracti32x4_epi32(v, 1));
        sum = _mm_add_epi32(sum, _mm512_extracti32x4_epi32(v, 2));
        sum = _mm_add_epi32(sum, _mm512_extracti32x4_epi32(v, 3));
        const __m128i hi64 = _mm_unpackhi_epi64(sum, sum);
        const __m128i sum64 = _mm_add_epi32(hi64, sum);
        const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        const __m128i sum32 = _mm_add_epi32(sum64, hi32);
        return _mm_cvtsi128_si32(sum32);
    }
}

i32 DotProductVnni(const i8* lhs, const i8* rhs, size_t length) noexcept {
    if (length < 64) {
        return DotProductAvx2(lhs, rhs, length);
    }

    const __m512i zero = _mm512_setzero_si512();
    const __m512i signBit = _mm512_set1_epi8(-128);
    const __m512i ones = _mm512_set1_epi8(1);

    __m512i dotProduct0 = zero;
    __m512i dotProduct1 = zero;
    __m512i dotProduct2 = zero;
    __m512i dotProduct3 = zero;

    while (length >= 256) {
        AccumulateDotProductI8Vnni(dotProduct0, lhs, rhs, zero, signBit, ones);
        AccumulateDotProductI8Vnni(dotProduct1, lhs + 64, rhs + 64, zero, signBit, ones);
        AccumulateDotProductI8Vnni(dotProduct2, lhs + 128, rhs + 128, zero, signBit, ones);
        AccumulateDotProductI8Vnni(dotProduct3, lhs + 192, rhs + 192, zero, signBit, ones);
        lhs += 256;
        rhs += 256;
        length -= 256;
    }

    while (length >= 64) {
        AccumulateDotProductI8Vnni(dotProduct0, lhs, rhs, zero, signBit, ones);
        lhs += 64;
        rhs += 64;
        length -= 64;
    }

    if (length > 0) {
        AccumulateTailDotProductI8Vnni(dotProduct0, lhs, rhs, length, zero, signBit, ones);
    }

    const __m512i dotProduct = _mm512_add_epi32(
        _mm512_add_epi32(dotProduct0, dotProduct1),
        _mm512_add_epi32(dotProduct2, dotProduct3));

    return HsumI32(dotProduct);
}

#else

i32 DotProductVnni(const i8* lhs, const i8* rhs, size_t length) noexcept {
    return DotProductAvx2(lhs, rhs, length);
}

#endif
