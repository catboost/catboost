#include <library/cpp/float16/float16.h>

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#include <util/system/cpu_id.h>
#include <util/system/yassert.h>

#include <cstring>

bool NFloat16Impl::AreConversionIntrinsicsAvailableOnHost() {
#ifdef _MSC_VER
    return false;
#else
    return NFloat16Ops::AreIntrinsicsAvailableOnHost();
#endif
}

ui16 NFloat16Impl::ConvertFloat32IntoFloat16Intrinsics(float val) {
#ifdef _MSC_VER
    Y_ABORT("MSVC doesn't have _cvtss_sh(), so NFloat16Impl::ConvertFloat32IntoFloat16Intrinsics() is not implemented");
#else
    return _cvtss_sh(val, _MM_FROUND_TO_NEAREST_INT);
#endif
}

float NFloat16Impl::ConvertFloat16IntoFloat32Intrinsics(ui16 val) {
#ifdef _MSC_VER
    Y_ABORT("MSVC doesn't have _cvtsh_ss(), so NFloat16Impl::ConvertFloat16IntoFloat32Intrinsics() is not implemented");
#else
    return _cvtsh_ss(val);
#endif
}

bool NFloat16Ops::AreIntrinsicsAvailableOnHost() {
    return NX86::CachedHaveF16C() && NX86::CachedHaveAVX();
}

void NFloat16Ops::UnpackFloat16SequenceIntrisincs(const TFloat16* src, float* dst, size_t len) {
    while (len >= 8) {
        __m128i source = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
        __m256 cvt = _mm256_cvtph_ps(source);

        _mm256_storeu_ps(dst, cvt);

        len -= 8;
        dst += 8;
        src += 8;
    }

    if (len > 0) {
        alignas(16) ui16 local[8] = {};
        memcpy(local, src, sizeof(*src) * len);
        __m128i source = _mm_load_si128(reinterpret_cast<const __m128i*>(local));
        __m256 cvt = _mm256_cvtph_ps(source);
        alignas(32) float localDst[8];
        _mm256_store_ps(localDst, cvt);

        memcpy(dst, localDst, len * sizeof(float));
    }
}

float NFloat16Ops::DotProductOnFloatIntrisincs(const float* f32, const TFloat16* f16, size_t len) {
    Y_ASSERT(size_t(f16) % Float16BufferAlignmentRequirementInBytes == 0);
    Y_ASSERT(size_t(f32) % Float32BufferAlignmentRequirementInBytes == 0);
    static_assert(sizeof(TFloat16) == 2, "unexpected TFloat16 size");

    __m256 sum = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 a, b, m;
    __m256 a2, b2, m2;
    __m128i source, source2;

    while (len >= 16) {
        source = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f16));
        source2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f16 + 8));
        b = _mm256_loadu_ps(f32);
        b2 = _mm256_loadu_ps(f32 + 8);
        a = _mm256_cvtph_ps(source);
        a2 = _mm256_cvtph_ps(source2);
        m = _mm256_mul_ps(a, b);
        m2 = _mm256_mul_ps(a2, b2);
        sum = _mm256_add_ps(sum, m);
        sum2 = _mm256_add_ps(sum2, m2);
        len -= 16;
        f16 += 16;
        f32 += 16;
    }

    if (len >= 8) {
        source = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f16));
        b = _mm256_loadu_ps(f32);
        a = _mm256_cvtph_ps(source);
        m = _mm256_mul_ps(a, b);
        sum = _mm256_add_ps(sum, m);
        len -= 8;
        f16 += 8;
        f32 += 8;
    }

    if (len > 0) {
        alignas(16) TFloat16 localF16[8] = {};
        alignas(32) float localF32[8] = {};

        memcpy(localF16, f16, sizeof(*f16) * len);
        memcpy(localF32, f32, sizeof(*f32) * len);

        __m128i source3 = _mm_load_si128(reinterpret_cast<const __m128i*>(localF16));
        a = _mm256_cvtph_ps(source3);
        b = _mm256_loadu_ps(localF32);
        m = _mm256_mul_ps(a, b);

        sum = _mm256_add_ps(sum, m);
    }

    sum = _mm256_add_ps(sum, sum2);

    alignas(32) float res[8];
    _mm256_store_ps(res, sum);
    return res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];
}

void NFloat16Ops::PackFloat16SequenceIntrisincs(const float* src, TFloat16* dst, size_t len) {
    while (len >= 8) {
        __m256 source = _mm256_loadu_ps(src);
        __m128i cvt = _mm256_cvtps_ph(source, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));

        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), cvt);

        len -= 8;
        dst += 8;
        src += 8;
    }

    if (len > 0) {
        alignas(32) float local[8] = {};
        memcpy(local, src, len * sizeof(float));
        __m256 source = _mm256_load_ps(local);
        __m128i cvt = _mm256_cvtps_ph(source, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        alignas(16) ui16 localDst[8];
        _mm_store_si128(reinterpret_cast<__m128i*>(localDst), cvt);

        memcpy(dst, localDst, len * sizeof(TFloat16));
    }
}
