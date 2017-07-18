#pragma once

#include "types.h"

#define Y_CPU_ID_ENUMERATE(F) \
    F(SSE)                    \
    F(SSE2)                   \
    F(SSE3)                   \
    F(SSSE3)                  \
    F(SSE41)                  \
    F(SSE42)                  \
    F(POPCNT)                 \
    F(BMI1)                   \
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

namespace NX86 {
    /**
     * returns false on non-x86 platforms
     */
    bool CpuId(ui32 op, ui32 res[4]) noexcept;
    bool CpuId(ui32 op, ui32 subOp, ui32 res[4]) noexcept;

#define Y_DEF_NAME(X) bool Have##X() noexcept;
    Y_CPU_ID_ENUMERATE(Y_DEF_NAME)
#undef Y_DEF_NAME

#define Y_DEF_NAME(X) bool CachedHave##X() noexcept;
    Y_CPU_ID_ENUMERATE(Y_DEF_NAME)
#undef Y_DEF_NAME
}

const char* CpuBrand(ui32 store[12]) noexcept;
