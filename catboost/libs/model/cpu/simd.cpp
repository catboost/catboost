#include "simd.h"

#include <catboost/libs/model/model.h>

#include <cmath>
#include <cstring>
#include <type_traits>

#include <util/system/cpu_id.h>
#include <util/system/platform.h>

#if defined(_x86_64_) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
    #include <immintrin.h>

namespace NCB::NModelEvaluation::NDetail {
    namespace {
        template <size_t Distance>
        Y_FORCE_INLINE void PrefetchLeafValues(size_t doc, size_t count, const double* const* leaves, const ui8* const* indexes) {
            if constexpr (Distance > 0) {
                if (doc + Distance < count) {
                    for (size_t tree = 0; tree < 4; ++tree) {
                        __builtin_prefetch(leaves[tree] + indexes[tree][doc + Distance], 0, 3);
                    }
                }
            }
        }

        template <size_t PrefetchDistance>
        __attribute__((target("avx2"))) void LeafValues4Avx2(
            size_t docCount,
            const double* const* leafValues,
            const ui8* const* indexes,
            double* results)
        {
            for (size_t doc = 0; doc < docCount; doc += 8) {
                PrefetchLeafValues<PrefetchDistance>(doc, docCount, leafValues, indexes);
                auto sum0 = _mm256_loadu_pd(results + doc);
                auto sum1 = _mm256_loadu_pd(results + doc + 4);
                for (size_t tree = 0; tree < 4; ++tree) {
                    const auto* leaves = leafValues[tree];
                    const auto* ids = indexes[tree] + doc;
                    const auto additions0 = _mm256_set_pd(
                        leaves[ids[3]], leaves[ids[2]], leaves[ids[1]], leaves[ids[0]]);
                    const auto additions1 = _mm256_set_pd(
                        leaves[ids[7]], leaves[ids[6]], leaves[ids[5]], leaves[ids[4]]);
                    sum0 = _mm256_add_pd(sum0, additions0);
                    sum1 = _mm256_add_pd(sum1, additions1);
                }
                _mm256_storeu_pd(results + doc, sum0);
                _mm256_storeu_pd(results + doc + 4, sum1);
            }
        }

        template <size_t PrefetchDistance>
        __attribute__((target("avx512f"))) void LeafValues4Avx512(
            size_t docCount,
            const double* const* leafValues,
            const ui8* const* indexes,
            double* results)
        {
            for (size_t doc = 0; doc < docCount; doc += 8) {
                PrefetchLeafValues<PrefetchDistance>(doc, docCount, leafValues, indexes);
                auto sum = _mm512_loadu_pd(results + doc);
                for (size_t tree = 0; tree < 4; ++tree) {
                    const auto* leaves = leafValues[tree];
                    const auto* ids = indexes[tree] + doc;
                    const auto additions = _mm512_set_pd(
                        leaves[ids[7]], leaves[ids[6]], leaves[ids[5]], leaves[ids[4]],
                        leaves[ids[3]], leaves[ids[2]], leaves[ids[1]], leaves[ids[0]]);
                    sum = _mm512_add_pd(sum, additions);
                }
                _mm512_storeu_pd(results + doc, sum);
            }
        }
    } // namespace

    TLeafValues4Function GetLeafValues4Avx2(size_t prefetchDistance) noexcept {
        if (!NX86::CachedHaveAVX2()) {
            return nullptr;
        }
        switch (prefetchDistance) {
            case 0:
                return LeafValues4Avx2<0>;
            case 8:
                return LeafValues4Avx2<8>;
            case 16:
                return LeafValues4Avx2<16>;
            case 32:
                return LeafValues4Avx2<32>;
            case 64:
                return LeafValues4Avx2<64>;
            default:
                return nullptr;
        }
    }

    TLeafValues4Function GetLeafValues4Avx512(size_t prefetchDistance) noexcept {
        if (!NX86::CachedHaveAVX512F()) {
            return nullptr;
        }
        switch (prefetchDistance) {
            case 0:
                return LeafValues4Avx512<0>;
            case 8:
                return LeafValues4Avx512<8>;
            case 16:
                return LeafValues4Avx512<16>;
            case 32:
                return LeafValues4Avx512<32>;
            case 64:
                return LeafValues4Avx512<64>;
            default:
                return nullptr;
        }
    }
} // namespace NCB::NModelEvaluation::NDetail
#else
namespace NCB::NModelEvaluation::NDetail {
    TLeafValues4Function GetLeafValues4Avx2(size_t) noexcept {
        return nullptr;
    }

    TLeafValues4Function GetLeafValues4Avx512(size_t) noexcept {
        return nullptr;
    }
} // namespace NCB::NModelEvaluation::NDetail
#endif

namespace NCB::NModelEvaluation::NDetail {
    TLeafValues4Function GetWideLeafValues4() noexcept {
        static const auto kernel = [] {
            if (auto avx512 = GetLeafValues4Avx512()) {
                return avx512;
            }
            return GetLeafValues4Avx2();
        }();
        return kernel;
    }
} // namespace NCB::NModelEvaluation::NDetail

#if defined(_x86_64_) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
namespace NCB::NModelEvaluation::NDetail {
    namespace {
        template <bool UseXor, size_t Width>
        __m128i PartialByteIndexes(const ui8* features, size_t count, size_t doc,
                                   const TRepackedBin* splits, int depth) {
            auto result = _mm_setzero_si128();
            auto bit = _mm_set1_epi8(1);
            for (int level = 0; level < depth; ++level) {
                const auto* source = features + splits[level].FeatureIndex * count + doc;
                __m128i value;
                if constexpr (Width == 16) {
                    value = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
                } else if constexpr (Width == 8) {
                    value = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source));
                } else {
                    ui32 packed;
                    std::memcpy(&packed, source, sizeof(packed));
                    value = _mm_cvtsi32_si128(packed);
                }
                if constexpr (UseXor) {
                    value = _mm_xor_si128(value, _mm_set1_epi8(splits[level].XorMask));
                }
                const auto border = _mm_set1_epi8(splits[level].SplitIdx);
                const auto ge = _mm_cmpeq_epi8(_mm_max_epu8(value, border), value);
                result = _mm_or_si128(result, _mm_and_si128(ge, bit));
                bit = _mm_slli_epi16(bit, 1);
            }
            return result;
        }

        template <bool UseXor>
        void ByteIndexesTail(const ui8* features, size_t count, size_t doc, ui8* result,
                             const TRepackedBin* splits, int depth) {
            for (; doc + 16 <= count; doc += 16) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(result + doc),
                                 PartialByteIndexes<UseXor, 16>(features, count, doc, splits, depth));
            }
            if (doc + 8 <= count) {
                _mm_storel_epi64(reinterpret_cast<__m128i*>(result + doc),
                                 PartialByteIndexes<UseXor, 8>(features, count, doc, splits, depth));
                doc += 8;
            }
            if (doc + 4 <= count) {
                const ui32 packed = _mm_cvtsi128_si32(PartialByteIndexes<UseXor, 4>(features, count, doc, splits, depth));
                std::memcpy(result + doc, &packed, sizeof(packed));
                doc += 4;
            }
            for (; doc < count; ++doc) {
                ui8 index = 0;
                for (int level = 0; level < depth; ++level) {
                    ui8 value = features[splits[level].FeatureIndex * count + doc];
                    if constexpr (UseXor) {
                        value ^= splits[level].XorMask;
                    }
                    index |= ui8(value >= ui8(splits[level].SplitIdx)) << level;
                }
                result[doc] = index;
            }
        }

        template <typename TIndex>
        void IndexesTail(const ui8* features, size_t count, size_t start, TIndex* result,
                         const TRepackedBin* splits, int depth, bool useXor) {
            if (start == count) {
                return;
            }
            if constexpr (std::is_same_v<TIndex, ui8>) {
                if (depth <= 8) {
                    if (useXor) {
                        ByteIndexesTail<true>(features, count, start, result, splits, depth);
                    } else {
                        ByteIndexesTail<false>(features, count, start, result, splits, depth);
                    }
                    return;
                }
            }
            for (size_t doc = start; doc < count; ++doc) {
                TIndex index = 0;
                for (int level = 0; level < depth; ++level) {
                    ui8 value = features[splits[level].FeatureIndex * count + doc];
                    if (useXor) {
                        value ^= splits[level].XorMask;
                    }
                    index |= TIndex(ui32(value >= ui8(splits[level].SplitIdx)) << level);
                }
                result[doc] = index;
            }
        }

        __attribute__((target("avx2"))) void TreeIndexes8Avx2(
            const ui8* features, size_t count, ui8* result, const TRepackedBin* splits, int depth, bool useXor) {
            size_t doc = 0;
            for (; doc + 32 <= count; doc += 32) {
                auto index = _mm256_setzero_si256();
                for (int level = 0; level < depth; ++level) {
                    auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(features + splits[level].FeatureIndex * count + doc));
                    if (useXor) {
                        value = _mm256_xor_si256(value, _mm256_set1_epi8(splits[level].XorMask));
                    }
                    const auto border = _mm256_set1_epi8(ui8(splits[level].SplitIdx));
                    const auto mask = _mm256_cmpeq_epi8(value, _mm256_max_epu8(value, border));
                    index = _mm256_or_si256(index, _mm256_and_si256(mask, _mm256_set1_epi8(ui8(1u << level))));
                }
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + doc), index);
            }
            IndexesTail(features, count, doc, result, splits, depth, useXor);
        }

        __attribute__((target("avx512f,avx512bw"))) void TreeIndexes8Avx512(
            const ui8* features, size_t count, ui8* result, const TRepackedBin* splits, int depth, bool useXor) {
            size_t doc = 0;
            for (; doc + 64 <= count; doc += 64) {
                auto index = _mm512_setzero_si512();
                for (int level = 0; level < depth; ++level) {
                    auto value = _mm512_loadu_si512(features + splits[level].FeatureIndex * count + doc);
                    if (useXor) {
                        value = _mm512_xor_si512(value, _mm512_set1_epi8(splits[level].XorMask));
                    }
                    const auto mask = _mm512_cmpge_epu8_mask(value, _mm512_set1_epi8(ui8(splits[level].SplitIdx)));
                    index = _mm512_or_si512(index, _mm512_maskz_mov_epi8(mask, _mm512_set1_epi8(ui8(1u << level))));
                }
                _mm512_storeu_si512(result + doc, index);
            }
            IndexesTail(features, count, doc, result, splits, depth, useXor);
        }

        __attribute__((target("avx2"))) void TreeIndexes32Avx2(
            const ui8* features, size_t count, ui32* result, const TRepackedBin* splits, int depth, bool useXor) {
            size_t doc = 0;
            for (; doc + 8 <= count; doc += 8) {
                auto index = _mm256_setzero_si256();
                for (int level = 0; level < depth; ++level) {
                    auto value = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(features + splits[level].FeatureIndex * count + doc)));
                    if (useXor) {
                        value = _mm256_xor_si256(value, _mm256_set1_epi32(splits[level].XorMask));
                    }
                    const auto mask = _mm256_cmpgt_epi32(value, _mm256_set1_epi32(int(ui8(splits[level].SplitIdx)) - 1));
                    index = _mm256_or_si256(index, _mm256_and_si256(mask, _mm256_set1_epi32(ui32(1) << level)));
                }
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + doc), index);
            }
            IndexesTail(features, count, doc, result, splits, depth, useXor);
        }

        __attribute__((target("avx512f"))) void TreeIndexes32Avx512(
            const ui8* features, size_t count, ui32* result, const TRepackedBin* splits, int depth, bool useXor) {
            size_t doc = 0;
            for (; doc + 16 <= count; doc += 16) {
                auto index = _mm512_setzero_si512();
                for (int level = 0; level < depth; ++level) {
                    auto value = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(features + splits[level].FeatureIndex * count + doc)));
                    if (useXor) {
                        value = _mm512_xor_si512(value, _mm512_set1_epi32(splits[level].XorMask));
                    }
                    const auto mask = _mm512_cmpge_epi32_mask(value, _mm512_set1_epi32(ui8(splits[level].SplitIdx)));
                    index = _mm512_or_si512(index, _mm512_maskz_mov_epi32(mask, _mm512_set1_epi32(ui32(1) << level)));
                }
                _mm512_storeu_si512(result + doc, index);
            }
            IndexesTail(features, count, doc, result, splits, depth, useXor);
        }

        void BinarizeTail(const float* values, size_t count, size_t start, const float* borders,
                          size_t borderCount, bool substitute, float replacement, ui8* result) {
            for (size_t doc = start; doc < count; ++doc) {
                float value = values[doc];
                if (substitute && std::isnan(value)) {
                    value = replacement;
                }
                for (size_t block = 0; block < borderCount; block += MAX_VALUES_PER_BIN) {
                    ui8 sum = 0;
                    for (size_t border = block; border < Min(block + MAX_VALUES_PER_BIN, borderCount); ++border) {
                        sum += value > borders[border];
                    }
                    auto& target = result[(block / MAX_VALUES_PER_BIN) * count + doc];
                    if (doc < count / 16 * 16) {
                        target = sum;
                    } else {
                        target += sum;
                    }
                }
            }
        }

        // A constant column only needs one comparison per border. Restrict
        // detection to wide, border-heavy calls so short/light columns retain
        // the existing path. Compare bits to preserve NaN and signed-zero cases.
        __attribute__((target("avx2"))) Y_NO_INLINE bool IsConstantAvx2(const float* values, size_t count) {
            ui32 first;
            std::memcpy(&first, values, sizeof(first));
            const auto expected = _mm256_set1_epi32(first);
            size_t doc = 0;
            for (; doc + 8 <= count; doc += 8) {
                const auto actual = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values + doc));
                if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(actual, expected)) != -1) {
                    return false;
                }
            }
            for (; doc < count; ++doc) {
                ui32 actual;
                std::memcpy(&actual, values + doc, sizeof(actual));
                if (actual != first) {
                    return false;
                }
            }
            return true;
        }

        __attribute__((target("avx512f"))) Y_NO_INLINE bool IsConstantAvx512(const float* values, size_t count) {
            ui32 first;
            std::memcpy(&first, values, sizeof(first));
            const auto expected = _mm512_set1_epi32(first);
            size_t doc = 0;
            for (; doc + 16 <= count; doc += 16) {
                if (_mm512_cmpeq_epi32_mask(_mm512_loadu_si512(values + doc), expected) != 0xffff) {
                    return false;
                }
            }
            for (; doc < count; ++doc) {
                ui32 actual;
                std::memcpy(&actual, values + doc, sizeof(actual));
                if (actual != first) {
                    return false;
                }
            }
            return true;
        }

        Y_NO_INLINE void BinarizeConstant(float value, size_t count, const float* borders, size_t borderCount,
                                          bool substitute, float replacement, ui8* result) {
            if (substitute && std::isnan(value)) {
                value = replacement;
            }
            const size_t vectorCount = count / 16 * 16;
            for (size_t block = 0; block < borderCount; block += MAX_VALUES_PER_BIN) {
                ui8 sum = 0;
                for (size_t border = block; border < Min(block + MAX_VALUES_PER_BIN, borderCount); ++border) {
                    sum += value > borders[border];
                }
                auto* target = result + (block / MAX_VALUES_PER_BIN) * count;
                std::memset(target, sum, vectorCount);
                // The existing vector path overwrites full groups of sixteen;
                // its scalar tail adds to the caller's buffer. Keep both semantics.
                for (size_t doc = vectorCount; doc < count; ++doc) {
                    target[doc] += sum;
                }
            }
        }

        __attribute__((target("avx2"))) Y_NO_INLINE void BinarizeAvx2Vector(
            const float* values, size_t count, const float* borders, size_t borderCount,
            bool substitute, float replacement, ui8* result) {
            size_t doc = 0;
            for (; doc + 8 <= count / 16 * 16; doc += 8) {
                auto value = _mm256_loadu_ps(values + doc);
                if (substitute) {
                    value = _mm256_blendv_ps(value, _mm256_set1_ps(replacement), _mm256_cmp_ps(value, value, _CMP_UNORD_Q));
                }
                for (size_t block = 0; block < borderCount; block += MAX_VALUES_PER_BIN) {
                    auto sum = _mm256_setzero_si256();
                    for (size_t border = block; border < Min(block + MAX_VALUES_PER_BIN, borderCount); ++border) {
                        const auto mask = _mm256_castps_si256(_mm256_cmp_ps(value, _mm256_set1_ps(borders[border]), _CMP_GT_OQ));
                        sum = _mm256_sub_epi32(sum, mask);
                    }
                    const auto packed16 = _mm_packus_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
                    const auto packed8 = _mm_packus_epi16(packed16, _mm_setzero_si128());
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(result + (block / MAX_VALUES_PER_BIN) * count + doc), packed8);
                }
            }
            BinarizeTail(values, count, doc, borders, borderCount, substitute, replacement, result);
        }

        __attribute__((target("avx512f"))) Y_NO_INLINE void BinarizeAvx512Vector(
            const float* values, size_t count, const float* borders, size_t borderCount,
            bool substitute, float replacement, ui8* result) {
            size_t doc = 0;
            for (; doc + 16 <= count; doc += 16) {
                auto value = _mm512_loadu_ps(values + doc);
                if (substitute) {
                    value = _mm512_mask_mov_ps(value, _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q), _mm512_set1_ps(replacement));
                }
                for (size_t block = 0; block < borderCount; block += MAX_VALUES_PER_BIN) {
                    auto sum = _mm512_setzero_si512();
                    const auto ones = _mm512_set1_epi32(1);
                    for (size_t border = block; border < Min(block + MAX_VALUES_PER_BIN, borderCount); ++border) {
                        const auto mask = _mm512_cmp_ps_mask(value, _mm512_set1_ps(borders[border]), _CMP_GT_OQ);
                        sum = _mm512_mask_add_epi32(sum, mask, sum, ones);
                    }
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(result + (block / MAX_VALUES_PER_BIN) * count + doc), _mm512_cvtusepi32_epi8(sum));
                }
            }
            BinarizeTail(values, count, doc, borders, borderCount, substitute, replacement, result);
        }

        Y_FORCE_INLINE bool MayBeConstant(const float* values, size_t count, size_t borderCount) {
            if (count < 64 || borderCount < 16) {
                return false;
            }
            ui32 first, last;
            std::memcpy(&first, values, sizeof(first));
            std::memcpy(&last, values + count - 1, sizeof(last));
            return first == last;
        }

        // Preserve all arguments only on the uncommon endpoint-match path.
        // The public dispatch wrapper can then reject variable columns without
        // saving registers or building a stack frame before its tail call.
        __attribute__((target("avx2"))) Y_NO_INLINE void BinarizePotentialConstantAvx2(
            const float* values, size_t count, const float* borders, size_t borderCount,
            bool substitute, float replacement, ui8* result) {
            if (IsConstantAvx2(values, count)) {
                BinarizeConstant(values[0], count, borders, borderCount, substitute, replacement, result);
                return;
            }
            BinarizeAvx2Vector(values, count, borders, borderCount, substitute, replacement, result);
        }

        __attribute__((target("avx512f"))) Y_NO_INLINE void BinarizePotentialConstantAvx512(
            const float* values, size_t count, const float* borders, size_t borderCount,
            bool substitute, float replacement, ui8* result) {
            if (IsConstantAvx512(values, count)) {
                BinarizeConstant(values[0], count, borders, borderCount, substitute, replacement, result);
                return;
            }
            BinarizeAvx512Vector(values, count, borders, borderCount, substitute, replacement, result);
        }

        __attribute__((target("avx2"))) void BinarizeAvx2(
            const float* values, size_t count, const float* borders, size_t borderCount,
            bool substitute, float replacement, ui8* result) {
            if (MayBeConstant(values, count, borderCount)) {
                BinarizePotentialConstantAvx2(values, count, borders, borderCount, substitute, replacement, result);
                return;
            }
            BinarizeAvx2Vector(values, count, borders, borderCount, substitute, replacement, result);
        }

        __attribute__((target("avx512f"))) void BinarizeAvx512(
            const float* values, size_t count, const float* borders, size_t borderCount,
            bool substitute, float replacement, ui8* result) {
            if (MayBeConstant(values, count, borderCount)) {
                BinarizePotentialConstantAvx512(values, count, borders, borderCount, substitute, replacement, result);
                return;
            }
            BinarizeAvx512Vector(values, count, borders, borderCount, substitute, replacement, result);
        }

    #if defined(__clang__)
        #pragma clang fp contract(off)
    #elif defined(__GNUC__)
        #pragma GCC push_options
        #pragma GCC optimize("fp-contract=off")
    #endif
        __attribute__((target("avx2"))) void ScaleBiasAvx2(double* values, size_t count, double scale, double bias, bool applyBias) {
            size_t index = 0;
            for (; index + 4 <= count; index += 4) {
                auto value = _mm256_loadu_pd(values + index);
                if (scale != 1.0) {
                    value = _mm256_mul_pd(value, _mm256_set1_pd(scale));
                }
                if (applyBias) {
                    value = _mm256_add_pd(value, _mm256_set1_pd(bias));
                }
                _mm256_storeu_pd(values + index, value);
            }
            for (; index < count; ++index) {
                if (scale != 1.0) {
                    values[index] *= scale;
                }
                if (applyBias) {
                    values[index] += bias;
                }
            }
        }
        __attribute__((target("avx512f"))) void ScaleBiasAvx512(double* values, size_t count, double scale, double bias, bool applyBias) {
            size_t index = 0;
            for (; index + 8 <= count; index += 8) {
                auto value = _mm512_loadu_pd(values + index);
                if (scale != 1.0) {
                    value = _mm512_mul_pd(value, _mm512_set1_pd(scale));
                }
                if (applyBias) {
                    value = _mm512_add_pd(value, _mm512_set1_pd(bias));
                }
                _mm512_storeu_pd(values + index, value);
            }
            for (; index < count; ++index) {
                if (scale != 1.0) {
                    values[index] *= scale;
                }
                if (applyBias) {
                    values[index] += bias;
                }
            }
        }
    } // namespace
} // namespace NCB::NModelEvaluation::NDetail
    #if defined(__GNUC__) && !defined(__clang__)
        #pragma GCC pop_options
    #endif
    #define DEFINE_KERNEL_GETTER(Name, Feature)                  \
        decltype(&Name) Get##Name() noexcept {                   \
            return NX86::CachedHave##Feature() ? Name : nullptr; \
        }
#else
    #define DEFINE_KERNEL_GETTER(Name, Feature)      \
        decltype(Get##Name()) Get##Name() noexcept { \
            return nullptr;                          \
        }
#endif

namespace NCB::NModelEvaluation::NDetail {
    DEFINE_KERNEL_GETTER(TreeIndexes8Avx2, AVX2)
    DEFINE_KERNEL_GETTER(TreeIndexes8Avx512, AVX512BW)
    DEFINE_KERNEL_GETTER(TreeIndexes32Avx2, AVX2)
    DEFINE_KERNEL_GETTER(TreeIndexes32Avx512, AVX512F)
    DEFINE_KERNEL_GETTER(BinarizeAvx2, AVX2)
    DEFINE_KERNEL_GETTER(BinarizeAvx512, AVX512F)
    DEFINE_KERNEL_GETTER(ScaleBiasAvx2, AVX2)
    DEFINE_KERNEL_GETTER(ScaleBiasAvx512, AVX512F)

#define DEFINE_WIDE_GETTER(Name, Type)               \
    Type GetWide##Name() noexcept {                  \
        static const auto kernel = [] {              \
            if (auto avx512 = Get##Name##Avx512()) { \
                return avx512;                       \
            }                                        \
            return Get##Name##Avx2();                \
        }();                                         \
        return kernel;                               \
    }
    DEFINE_WIDE_GETTER(TreeIndexes8, TTreeIndexesFunction<ui8>)
    DEFINE_WIDE_GETTER(TreeIndexes32, TTreeIndexesFunction<ui32>)
    DEFINE_WIDE_GETTER(Binarize, TBinarizeFunction)
    DEFINE_WIDE_GETTER(ScaleBias, TScaleBiasFunction)
#undef DEFINE_WIDE_GETTER
} // namespace NCB::NModelEvaluation::NDetail
#undef DEFINE_KERNEL_GETTER
