#include <library/float16/float16.h>

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#include <util/system/cpu_id.h>
#include <util/system/yassert.h>
#include <util/system/sanitizers.h>

bool NFloat16Ops::IsIntrisincsAvailableOnHost() {
    return NX86::CachedHaveF16C() && NX86::CachedHaveAVX();
}


void NFloat16Ops::UnpackFloat16SequenceIntrisincs(const TFloat16* src, float* dst, size_t len) {
    Y_ASSERT(size_t(src) % Float16BufferAlignmentRequirementInBytes == 0);

    while (len >= 8) {
        __m128i source = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
        __m256 cvt = _mm256_cvtph_ps(source);

        _mm256_store_ps(dst, cvt);

        len -= 8;
        dst += 8;
        src += 8;
    }

    if (len > 0) {
        alignas(16) ui16 local[8];
        NSan::Unpoison(local, sizeof(local));
        memcpy(local, src, sizeof(*src) * len);
        __m128i source = _mm_loadu_si128(reinterpret_cast<const __m128i*>(local));
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
        alignas(16) TFloat16 localF16[8];
        alignas(32) float localF32[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        NSan::Unpoison(localF16, sizeof(localF16));
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
