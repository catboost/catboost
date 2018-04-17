#pragma once

#include "defaults.h"

//#define USE_GENERIC_ENDIAN_CVT

#if defined(_linux_) && !defined(USE_GENERIC_ENDIAN_CVT)
#include <byteswap.h>
#elif defined(_darwin_)
#if defined(_arm_) || defined(__IOS__)
#include <architecture/byte_order.h>
#else
#include <machine/byte_order.h>
#endif
#else
#include <util/generic/utility.h>
#endif

#if defined(_linux_) && !defined(USE_GENERIC_ENDIAN_CVT)
#define SwapBytes16 bswap_16
#define SwapBytes32 bswap_32
#define SwapBytes64 bswap_64
#elif defined(_darwin_)
#ifdef _arm_
#define SwapBytes16 _OSSwapInt16
#define SwapBytes32 _OSSwapInt32
#define SwapBytes64 _OSSwapInt64
#else
#define SwapBytes16 OSSwapInt16
#define SwapBytes32 OSSwapInt32
#define SwapBytes64 OSSwapInt64
#endif
#else
#if defined(_x86_) && defined(__GNUC__)
#undef asm
#define asm __asm__

inline ui16 SwapBytes16(ui16 x) noexcept {
    ui16 val;

    asm(
        "rorw $8, %w0"
        : "=r"(val)
        : "0"(x)
        : "cc");

    return val;
}

inline ui32 SwapBytes32(ui32 x) noexcept {
    ui32 val;

    asm(
        "bswap %0"
        : "=r"(val)
        : "0"(x));

    return val;
}

#if defined(_x86_64_)
#define HAVE_SWAP_BYTES_64

inline ui64 SwapBytes64(ui64 x) noexcept {
    ui64 val;

    asm(
        "bswapq %0"
        : "=r"(val)
        : "0"(x));

    return val;
}
#endif

#undef asm
#else
#define byte_n(__val, __n) ((((unsigned char*)(&__val))[__n]))

inline ui16 SwapBytes16(ui16 val) noexcept {
    DoSwap(byte_n(val, 0), byte_n(val, 1));

    return val;
}

inline ui32 SwapBytes32(ui32 val) noexcept {
    DoSwap(byte_n(val, 0), byte_n(val, 3));
    DoSwap(byte_n(val, 1), byte_n(val, 2));

    return val;
}

#undef byte_n
#endif

#if !defined(HAVE_SWAP_BYTES_64)
inline ui64 SwapBytes64(ui64 val) noexcept {
    union {
        ui64 val;
        ui32 p[2];
    } tmp, ret;

    tmp.val = val;
    ret.p[0] = SwapBytes32(tmp.p[1]);
    ret.p[1] = SwapBytes32(tmp.p[0]);

    return ret.val;
}
#endif

#undef HAVE_SWAP_BYTES_64
#endif

//for convenience
static inline ui8 SwapBytes8(ui8 v) noexcept {
    return v;
}

namespace NSwapBytes {
    template <unsigned N>
    struct TSwapBytesHelper {
    };

#define DEF_SB(X)                             \
    template <>                               \
    struct TSwapBytesHelper<X> {              \
        template <class T>                    \
        static inline T Swap(T t) noexcept {  \
            return (T)SwapBytes##X((ui##X)t); \
        }                                     \
    };

    DEF_SB(8)
    DEF_SB(16)
    DEF_SB(32)
    DEF_SB(64)

#undef DEF_SB
}

template <class T>
inline T SwapBytes(T val) noexcept {
    return NSwapBytes::TSwapBytesHelper<sizeof(T) * 8>::Swap(val);
}

template <class T>
inline T LittleToBig(T val) noexcept {
    return SwapBytes(val);
}

template <class T>
inline T BigToLittle(T val) noexcept {
    return LittleToBig(val);
}

template <class T>
inline T HostToInet(T val) noexcept {
#if defined(_big_endian_)
    return val;
#elif defined(_little_endian_)
    return LittleToBig(val);
#else
#error todo
#endif
}

template <class T>
inline T InetToHost(T val) noexcept {
    return HostToInet(val);
}

template <class T>
inline T HostToLittle(T val) noexcept {
#if defined(_big_endian_)
    return BigToLittle(val);
#elif defined(_little_endian_)
    return val;
#else
#error todo
#endif
}

template <class T>
inline T LittleToHost(T val) noexcept {
    return HostToLittle(val);
}
