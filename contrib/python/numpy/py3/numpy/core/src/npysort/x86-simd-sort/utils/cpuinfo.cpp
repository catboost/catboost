/*******************************************
 * * Copyright (C) 2022 Intel Corporation
 * * SPDX-License-Identifier: BSD-3-Clause
 * *******************************************/

#include "cpuinfo.h"

static void
cpuid(uint32_t feature, uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d)
{
    __asm__ volatile(
            "cpuid"
            "\n\t"
            : "=a"(*a), "=b"(*b), "=c"(*c), "=d"(*d)
            : "a"(feature), "c"(0));
}

int cpu_has_avx512_vbmi2()
{
    uint32_t eax(0), ebx(0), ecx(0), edx(0);
    cpuid(0x07, &eax, &ebx, &ecx, &edx);
    return (ecx >> 6) & 0x1;
}

int cpu_has_avx512bw()
{
    uint32_t eax(0), ebx(0), ecx(0), edx(0);
    cpuid(0x07, &eax, &ebx, &ecx, &edx);
    return (ebx >> 30) & 0x1;
}

int cpu_has_avx512fp16()
{
    uint32_t eax(0), ebx(0), ecx(0), edx(0);
    cpuid(0x07, &eax, &ebx, &ecx, &edx);
    return (edx >> 23) & 0x1;
}

// TODO:
//int check_os_supports_avx512()
//{
//    uint32_t eax(0), ebx(0), ecx(0), edx(0);
//    cpuid(0x01, &eax, &ebx, &ecx, &edx);
//    // XSAVE:
//    if ((ecx >> 27) & 0x1) {
//	uint32_t xget_eax, xget_edx, index(0);
//	__asm__ ("xgetbv" : "=a"(xget_eax), "=d"(xget_edx) : "c" (index))
//    }
//
//}
