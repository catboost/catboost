#include "onehot_simd.h"

#include <catboost/libs/model/model.h>

#include <util/system/cpu_id.h>
#include <util/system/platform.h>

#if defined(_x86_64_) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
    #include <immintrin.h>

namespace NCB::NModelEvaluation::NDetail {
    namespace {
        void OneHotTail(const ui32* hashes, size_t count, size_t start, const int* values,
                        size_t valueCount, ui8* result) {
            for (size_t doc = start; doc < count; ++doc) {
                for (size_t block = 0; block < valueCount; block += MAX_VALUES_PER_BIN) {
                    auto& target = result[(block / MAX_VALUES_PER_BIN) * count + doc];
                    for (size_t value = block; value < Min(block + MAX_VALUES_PER_BIN, valueCount); ++value) {
                        target |= ui8(hashes[doc] == ui32(values[value])) * (value - block + 1);
                    }
                }
            }
        }
        __attribute__((target("avx2"))) void OneHotAvx2(
            const ui32* hashes, size_t count, const int* values, size_t valueCount, ui8* result) {
            size_t doc = 0;
            for (; doc + 8 <= count; doc += 8) {
                const auto hashesVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hashes + doc));
                for (size_t block = 0; block < valueCount; block += MAX_VALUES_PER_BIN) {
                    auto* output = result + (block / MAX_VALUES_PER_BIN) * count + doc;
                    auto codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(output)));
                    for (size_t value = block; value < Min(block + MAX_VALUES_PER_BIN, valueCount); ++value) {
                        const auto equal = _mm256_cmpeq_epi32(hashesVec, _mm256_set1_epi32(values[value]));
                        codes = _mm256_or_si256(codes, _mm256_and_si256(equal, _mm256_set1_epi32(value - block + 1)));
                    }
                    const auto packed16 = _mm_packus_epi32(_mm256_castsi256_si128(codes), _mm256_extracti128_si256(codes, 1));
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(output), _mm_packus_epi16(packed16, _mm_setzero_si128()));
                }
            }
            OneHotTail(hashes, count, doc, values, valueCount, result);
        }
        __attribute__((target("avx512f"))) void OneHotAvx512(
            const ui32* hashes, size_t count, const int* values, size_t valueCount, ui8* result) {
            size_t doc = 0;
            for (; doc + 16 <= count; doc += 16) {
                const auto hashesVec = _mm512_loadu_si512(hashes + doc);
                for (size_t block = 0; block < valueCount; block += MAX_VALUES_PER_BIN) {
                    auto* output = result + (block / MAX_VALUES_PER_BIN) * count + doc;
                    auto codes = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(output)));
                    for (size_t value = block; value < Min(block + MAX_VALUES_PER_BIN, valueCount); ++value) {
                        const auto equal = _mm512_cmpeq_epi32_mask(hashesVec, _mm512_set1_epi32(values[value]));
                        codes = _mm512_or_si512(codes, _mm512_maskz_mov_epi32(equal, _mm512_set1_epi32(value - block + 1)));
                    }
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(output), _mm512_cvtusepi32_epi8(codes));
                }
            }
            OneHotTail(hashes, count, doc, values, valueCount, result);
        }
    } // namespace
    TOneHotFunction GetOneHotAvx2() noexcept {
        return NX86::CachedHaveAVX2() ? OneHotAvx2 : nullptr;
    }
    TOneHotFunction GetOneHotAvx512() noexcept {
        return NX86::CachedHaveAVX512F() ? OneHotAvx512 : nullptr;
    }
} // namespace NCB::NModelEvaluation::NDetail
#else
namespace NCB::NModelEvaluation::NDetail {
    TOneHotFunction GetOneHotAvx2() noexcept {
        return nullptr;
    }
    TOneHotFunction GetOneHotAvx512() noexcept {
        return nullptr;
    }
} // namespace NCB::NModelEvaluation::NDetail
#endif

namespace NCB::NModelEvaluation::NDetail {
    TOneHotFunction GetWideOneHot() noexcept {
        static const auto kernel = [] {
            if (auto avx512 = GetOneHotAvx512()) {
                return avx512;
            }
            return GetOneHotAvx2();
        }();
        return kernel;
    }
} // namespace NCB::NModelEvaluation::NDetail
