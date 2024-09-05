#pragma once

#include "types.h"
#include "compiler.h"
#include <util/generic/singleton.h>

#define Y_CPU_ID_ENUMERATE(F) \
    F(SSE)                    \
    F(SSE2)                   \
    F(SSE3)                   \
    F(SSSE3)                  \
    F(SSE41)                  \
    F(SSE42)                  \
    F(F16C)                   \
    F(POPCNT)                 \
    F(BMI1)                   \
    F(BMI2)                   \
    F(PCLMUL)                 \
    F(AES)                    \
    F(AVX)                    \
    F(FMA)                    \
    F(AVX2)                   \
    F(AVX512F)                \
    F(AVX512DQ)               \
    F(AVX512IFMA)             \
    F(AVX512PF)               \
    F(AVX512ER)               \
    F(AVX512CD)               \
    F(AVX512BW)               \
    F(AVX512VL)               \
    F(AVX512VBMI)             \
    F(PREFETCHWT1)            \
    F(SHA)                    \
    F(ADX)                    \
    F(RDRAND)                 \
    F(RDSEED)                 \
    F(PCOMMIT)                \
    F(RDTSCP)                 \
    F(CLFLUSHOPT)             \
    F(CLWB)                   \
    F(XSAVE)                  \
    F(OSXSAVE)

#define Y_CPU_ID_ENUMERATE_OUTLINED_CACHED_DEFINE(F) \
    F(F16C)                                          \
    F(BMI1)                                          \
    F(BMI2)                                          \
    F(PCLMUL)                                        \
    F(AES)                                           \
    F(AVX)                                           \
    F(FMA)                                           \
    F(AVX2)                                          \
    F(AVX512F)                                       \
    F(AVX512DQ)                                      \
    F(AVX512IFMA)                                    \
    F(AVX512PF)                                      \
    F(AVX512ER)                                      \
    F(AVX512CD)                                      \
    F(AVX512BW)                                      \
    F(AVX512VL)                                      \
    F(AVX512VBMI)                                    \
    F(PREFETCHWT1)                                   \
    F(SHA)                                           \
    F(ADX)                                           \
    F(RDRAND)                                        \
    F(RDSEED)                                        \
    F(PCOMMIT)                                       \
    F(RDTSCP)                                        \
    F(CLFLUSHOPT)                                    \
    F(CLWB)                                          \
    F(XSAVE)                                         \
    F(OSXSAVE)

namespace NX86 {
    /**
     * returns false on non-x86 platforms
     */
    bool CpuId(ui32 op, ui32 res[4]) noexcept;
    bool CpuId(ui32 op, ui32 subOp, ui32 res[4]) noexcept;

#define Y_DEF_NAME(X) Y_CONST_FUNCTION bool Have##X() noexcept;
    Y_CPU_ID_ENUMERATE(Y_DEF_NAME)
#undef Y_DEF_NAME

#define Y_DEF_NAME(X) Y_CONST_FUNCTION bool CachedHave##X() noexcept;
    Y_CPU_ID_ENUMERATE_OUTLINED_CACHED_DEFINE(Y_DEF_NAME)
#undef Y_DEF_NAME

    struct TFlagsCache {
#define Y_DEF_NAME(X) const bool Have##X##_ = NX86::Have##X();
        Y_CPU_ID_ENUMERATE(Y_DEF_NAME)
#undef Y_DEF_NAME
    };

#define Y_LOOKUP_CPU_ID_IMPL(X) return SingletonWithPriority<TFlagsCache, 0>()->Have##X##_;

    inline bool CachedHaveSSE() noexcept {
#ifdef _sse_
        return true;
#else
        Y_LOOKUP_CPU_ID_IMPL(SSE)
#endif
    }

    inline bool CachedHaveSSE2() noexcept {
#ifdef _sse2_
        return true;
#else
        Y_LOOKUP_CPU_ID_IMPL(SSE2)
#endif
    }

    inline bool CachedHaveSSE3() noexcept {
#ifdef _sse3_
        return true;
#else
        Y_LOOKUP_CPU_ID_IMPL(SSE3)
#endif
    }

    inline bool CachedHaveSSSE3() noexcept {
#ifdef _ssse3_
        return true;
#else
        Y_LOOKUP_CPU_ID_IMPL(SSSE3)
#endif
    }

    inline bool CachedHaveSSE41() noexcept {
#ifdef _sse4_1_
        return true;
#else
        Y_LOOKUP_CPU_ID_IMPL(SSE41)
#endif
    }

    inline bool CachedHaveSSE42() noexcept {
#ifdef _sse4_2_
        return true;
#else
        Y_LOOKUP_CPU_ID_IMPL(SSE42)
#endif
    }

    inline bool CachedHavePOPCNT() noexcept {
#ifdef _popcnt_
        return true;
#else
        Y_LOOKUP_CPU_ID_IMPL(POPCNT)
#endif
    }

#undef Y_LOOKUP_CPU_ID_IMPL

} // namespace NX86

const char* CpuBrand(ui32 store[12]) noexcept;
