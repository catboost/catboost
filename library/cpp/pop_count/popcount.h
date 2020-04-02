#pragma once

#include <util/generic/typelist.h>
#include <util/system/cpu_id.h>
#include <util/system/defaults.h>
#include <util/system/hi_lo.h>
#include <util/system/platform.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

static inline ui32 PopCountImpl(ui8 n) {
#if defined(_ppc64_)
    ui32 r;
    __asm__("popcntb %0, %1"
            : "=r"(r)
            : "r"(n)
            :);
    return r;
#else
    extern ui8 const* PopCountLUT8;
    return PopCountLUT8[n];
#endif
}

static inline ui32 PopCountImpl(ui16 n) {
#if defined(_MSC_VER)
    return __popcnt16(n);
#else
    extern ui8 const* PopCountLUT16;
    return PopCountLUT16[n];
#endif
}

static inline ui32 PopCountImpl(ui32 n) {
#if defined(_MSC_VER)
    return __popcnt(n);
#else
#if defined(_x86_64_)
    if (NX86::CachedHavePOPCNT()) {
        ui32 r;

        __asm__("popcnt %1, %0;"
                : "=r"(r)
                : "r"(n)
                :);

        return r;
    }
#else
#if defined(_ppc64_)
    ui32 r;

    __asm__("popcntw %0, %1"
            : "=r"(r)
            : "r"(n)
            :);

    return r;
#endif
#endif

    return PopCountImpl((ui16)Lo16(n)) + PopCountImpl((ui16)Hi16(n));
#endif
}

static inline ui32 PopCountImpl(ui64 n) {
#if defined(_MSC_VER) && !defined(_i386_)
    return __popcnt64(n);
#else
#if defined(_x86_64_)
    if (NX86::CachedHavePOPCNT()) {
        ui64 r;

        __asm__("popcnt %1, %0;"
                : "=r"(r)
                : "r"(n)
                :);

        return r;
    }
#else
#if defined(_ppc64_)
    ui32 r;

    __asm__("popcntd %0, %1"
            : "=r"(r)
            : "r"(n)
            :);

    return r;
#endif
#endif

    return PopCountImpl((ui32)Lo32(n)) + PopCountImpl((ui32)Hi32(n));
#endif
}

template <class T>
static inline ui32 PopCount(T n) {
    using TCvt = TFixedWidthUnsignedInt<T>;

    return PopCountImpl((TCvt)n);
}
