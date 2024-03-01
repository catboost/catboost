/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_CPUID_HPP
#define XSIMD_CPUID_HPP

#include <algorithm>
#include <cstring>

#if defined(__linux__) && (defined(__ARM_NEON) || defined(_M_ARM) || defined(__riscv_vector))
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#if defined(_MSC_VER)
// Contains the definition of __cpuidex
#include <intrin.h>
#endif

#include "../types/xsimd_all_registers.hpp"

namespace xsimd
{
    namespace detail
    {
        struct supported_arch
        {
            unsigned sse2 : 1;
            unsigned sse3 : 1;
            unsigned ssse3 : 1;
            unsigned sse4_1 : 1;
            unsigned sse4_2 : 1;
            unsigned sse4a : 1;
            unsigned fma3_sse : 1;
            unsigned fma4 : 1;
            unsigned xop : 1;
            unsigned avx : 1;
            unsigned fma3_avx : 1;
            unsigned avx2 : 1;
            unsigned avxvnni : 1;
            unsigned fma3_avx2 : 1;
            unsigned avx512f : 1;
            unsigned avx512cd : 1;
            unsigned avx512dq : 1;
            unsigned avx512bw : 1;
            unsigned avx512er : 1;
            unsigned avx512pf : 1;
            unsigned avx512ifma : 1;
            unsigned avx512vbmi : 1;
            unsigned avx512vnni_bw : 1;
            unsigned avx512vnni_vbmi : 1;
            unsigned neon : 1;
            unsigned neon64 : 1;
            unsigned sve : 1;
            unsigned rvv : 1;

            // version number of the best arch available
            unsigned best;

            inline supported_arch() noexcept
            {
                memset(this, 0, sizeof(supported_arch));

#if defined(__aarch64__) || defined(_M_ARM64)
                neon = 1;
                neon64 = 1;
                best = neon64::version();
#elif defined(__ARM_NEON) || defined(_M_ARM)

#if defined(__linux__) && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 18)
                neon = bool(getauxval(AT_HWCAP) & HWCAP_NEON);
#else
                // that's very conservative :-/
                neon = 0;
#endif
                neon64 = 0;
                best = neon::version() * neon;

#elif defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0

#if defined(__linux__) && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 18)
                sve = bool(getauxval(AT_HWCAP) & HWCAP_SVE);
#else
                sve = 0;
#endif
                best = sve::version() * sve;

#elif defined(__riscv_vector) && defined(__riscv_v_fixed_vlen) && __riscv_v_fixed_vlen > 0

#if defined(__linux__) && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 18)
#ifndef HWCAP_V
#define HWCAP_V (1 << ('V' - 'A'))
#endif
                rvv = bool(getauxval(AT_HWCAP) & HWCAP_V);
#else
                rvv = 0;
#endif

                best = ::xsimd::rvv::version() * rvv;
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_AMD64) || defined(_M_IX86)
                auto get_cpuid = [](int reg[4], int level, int count = 0) noexcept
                {

#if defined(_MSC_VER)
                    __cpuidex(reg, level, count);

#elif defined(__INTEL_COMPILER)
                    __cpuid(reg, level);

#elif defined(__GNUC__) || defined(__clang__)

#if defined(__i386__) && defined(__PIC__)
                    // %ebx may be the PIC register
                    __asm__("xchg{l}\t{%%}ebx, %1\n\t"
                            "cpuid\n\t"
                            "xchg{l}\t{%%}ebx, %1\n\t"
                            : "=a"(reg[0]), "=r"(reg[1]), "=c"(reg[2]),
                              "=d"(reg[3])
                            : "0"(level), "2"(count));

#else
                    __asm__("cpuid\n\t"
                            : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]),
                              "=d"(reg[3])
                            : "0"(level), "2"(count));
#endif

#else
#error "Unsupported configuration"
#endif
                };

                int regs1[4];

                get_cpuid(regs1, 0x1);

                sse2 = regs1[3] >> 26 & 1;
                best = std::max(best, sse2::version() * sse2);

                sse3 = regs1[2] >> 0 & 1;
                best = std::max(best, sse3::version() * sse3);

                ssse3 = regs1[2] >> 9 & 1;
                best = std::max(best, ssse3::version() * ssse3);

                sse4_1 = regs1[2] >> 19 & 1;
                best = std::max(best, sse4_1::version() * sse4_1);

                sse4_2 = regs1[2] >> 20 & 1;
                best = std::max(best, sse4_2::version() * sse4_2);

                fma3_sse = regs1[2] >> 12 & 1;
                if (sse4_2)
                    best = std::max(best, fma3<xsimd::sse4_2>::version() * fma3_sse);

                avx = regs1[2] >> 28 & 1;
                best = std::max(best, avx::version() * avx);

                fma3_avx = avx && fma3_sse;
                best = std::max(best, fma3<xsimd::avx>::version() * fma3_avx);

                int regs8[4];
                get_cpuid(regs8, 0x80000001);
                fma4 = regs8[2] >> 16 & 1;
                best = std::max(best, fma4::version() * fma4);

                // sse4a = regs[2] >> 6 & 1;
                // best = std::max(best, XSIMD_X86_AMD_SSE4A_VERSION * sse4a);

                // xop = regs[2] >> 11 & 1;
                // best = std::max(best, XSIMD_X86_AMD_XOP_VERSION * xop);

                int regs7[4];
                get_cpuid(regs7, 0x7);
                avx2 = regs7[1] >> 5 & 1;
                best = std::max(best, avx2::version() * avx2);

                int regs7a[4];
                get_cpuid(regs7a, 0x7, 0x1);
                avxvnni = regs7a[0] >> 4 & 1;
                best = std::max(best, avxvnni::version() * avxvnni * avx2);

                fma3_avx2 = avx2 && fma3_sse;
                best = std::max(best, fma3<xsimd::avx2>::version() * fma3_avx2);

                avx512f = regs7[1] >> 16 & 1;
                best = std::max(best, avx512f::version() * avx512f);

                avx512cd = regs7[1] >> 28 & 1;
                best = std::max(best, avx512cd::version() * avx512cd * avx512f);

                avx512dq = regs7[1] >> 17 & 1;
                best = std::max(best, avx512dq::version() * avx512dq * avx512cd * avx512f);

                avx512bw = regs7[1] >> 30 & 1;
                best = std::max(best, avx512bw::version() * avx512bw * avx512dq * avx512cd * avx512f);

                avx512er = regs7[1] >> 27 & 1;
                best = std::max(best, avx512er::version() * avx512er * avx512cd * avx512f);

                avx512pf = regs7[1] >> 26 & 1;
                best = std::max(best, avx512pf::version() * avx512pf * avx512er * avx512cd * avx512f);

                avx512ifma = regs7[1] >> 21 & 1;
                best = std::max(best, avx512ifma::version() * avx512ifma * avx512bw * avx512dq * avx512cd * avx512f);

                avx512vbmi = regs7[2] >> 1 & 1;
                best = std::max(best, avx512vbmi::version() * avx512vbmi * avx512ifma * avx512bw * avx512dq * avx512cd * avx512f);

                avx512vnni_bw = regs7[2] >> 11 & 1;
                best = std::max(best, avx512vnni<xsimd::avx512bw>::version() * avx512vnni_bw * avx512bw * avx512dq * avx512cd * avx512f);

                avx512vnni_vbmi = avx512vbmi && avx512vnni_bw;
                best = std::max(best, avx512vnni<xsimd::avx512vbmi>::version() * avx512vnni_vbmi);
#endif
            }
        };
    }

    inline detail::supported_arch available_architectures() noexcept
    {
        static detail::supported_arch supported;
        return supported;
    }
}

#endif
