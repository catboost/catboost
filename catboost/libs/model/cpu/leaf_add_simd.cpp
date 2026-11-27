#include "leaf_add_simd.h"

#include <util/system/cpu_id.h>
#include <util/system/platform.h>

#if defined(_x86_64_) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
    #include <immintrin.h>
namespace NCB::NModelEvaluation::NDetail {
    namespace {
        template <typename TIndex>
        __attribute__((target("avx2"))) void AddLeafAvx2(size_t count, const double* leaves, const TIndex* indexes, double* result) {
            size_t doc = 0;
            for (; doc + 4 <= count; doc += 4) {
                const auto values = _mm256_set_pd(leaves[indexes[doc + 3]], leaves[indexes[doc + 2]], leaves[indexes[doc + 1]], leaves[indexes[doc]]);
                _mm256_storeu_pd(result + doc, _mm256_add_pd(_mm256_loadu_pd(result + doc), values));
            }
            for (; doc < count; ++doc) {
                result[doc] += leaves[indexes[doc]];
            }
        }
        template <typename TIndex>
        __attribute__((target("avx512f"))) void AddLeafAvx512(size_t count, const double* leaves, const TIndex* indexes, double* result) {
            size_t doc = 0;
            for (; doc + 8 <= count; doc += 8) {
                const auto values = _mm512_set_pd(leaves[indexes[doc + 7]], leaves[indexes[doc + 6]], leaves[indexes[doc + 5]], leaves[indexes[doc + 4]],
                                                  leaves[indexes[doc + 3]], leaves[indexes[doc + 2]], leaves[indexes[doc + 1]], leaves[indexes[doc]]);
                _mm512_storeu_pd(result + doc, _mm512_add_pd(_mm512_loadu_pd(result + doc), values));
            }
            for (; doc < count; ++doc) {
                result[doc] += leaves[indexes[doc]];
            }
        }
    } // namespace
} // namespace NCB::NModelEvaluation::NDetail
    #define ADD_LEAF_GETTER(Name, Type, Index, Feature, Impl)           \
        Type<Index> Get##Name() noexcept {                              \
            return NX86::CachedHave##Feature() ? Impl<Index> : nullptr; \
        }
#else
    #define ADD_LEAF_GETTER(Name, Type, Index, Feature, Impl) \
        Type<Index> Get##Name() noexcept {                    \
            return nullptr;                                   \
        }
#endif
namespace NCB::NModelEvaluation::NDetail {
    ADD_LEAF_GETTER(AddLeaf8Avx2, TAddLeafFunction, ui8, AVX2, AddLeafAvx2)
    ADD_LEAF_GETTER(AddLeaf8Avx512, TAddLeafFunction, ui8, AVX512F, AddLeafAvx512)
    ADD_LEAF_GETTER(AddLeaf32Avx2, TAddLeafFunction, ui32, AVX2, AddLeafAvx2)
    ADD_LEAF_GETTER(AddLeaf32Avx512, TAddLeafFunction, ui32, AVX512F, AddLeafAvx512)
#undef ADD_LEAF_GETTER
#define WIDE_LEAF_GETTER(Name, Type, Index)          \
    Type<Index> GetWide##Name() noexcept {           \
        static const auto kernel = [] {              \
            if (auto avx512 = Get##Name##Avx512()) { \
                return avx512;                       \
            }                                        \
            return Get##Name##Avx2();                \
        }();                                         \
        return kernel;                               \
    }
    WIDE_LEAF_GETTER(AddLeaf8, TAddLeafFunction, ui8)
    WIDE_LEAF_GETTER(AddLeaf32, TAddLeafFunction, ui32)
#undef WIDE_LEAF_GETTER
} // namespace NCB::NModelEvaluation::NDetail
