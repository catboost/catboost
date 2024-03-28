#pragma once

// ya style breaks indentation in ifdef's and code becomes unreadable
// clang-format off

// What OS ?
// our definition has the form _{osname}_

#if defined(_WIN64)
    #define _win64_
    #define _win32_
#elif defined(__WIN32__) || defined(_WIN32) // _WIN32 is also defined by the 64-bit compiler for backward compatibility
    #define _win32_
#else
    #define _unix_

    #if defined(__sun__) || defined(sun) || defined(sparc) || defined(__sparc)
        #define _sun_
    #endif

    #if defined(__hpux__)
        #define _hpux_
    #endif

    #if defined(__linux__)
        // Stands for "Linux" in the means of Linux kernel (i. e. Android is included)
        #define _linux_
    #endif

    #if defined(__FreeBSD__)
        #define _freebsd_
    #endif

    #if defined(__CYGWIN__)
        #define _cygwin_
    #endif

    #if defined(__APPLE__)
        #define _darwin_
    #endif

    #if defined(__ANDROID__)
        #define _android_
    #endif

    #if defined(__EMSCRIPTEN__)
        #define _emscripten_
    #endif
#endif

#if defined(__IOS__)
    #define _ios_
#endif

#if defined(_linux_)
    #if defined(_musl_)
        // nothing to do
    #elif defined(_android_)
        // Please do not mix with android-based systems.
        // This definition describes Standard Library (libc) type.
        #define _bionic_
    #else
        #define _glibc_
    #endif
#endif

#if defined(_darwin_)
    #define unix
    #define __unix__
#endif

#if defined(_win32_) || defined(_win64_)
    #define _win_
#endif

#if defined(__arm__) || defined(__ARM__) || defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM)
    #if defined(__arm64) || defined(__arm64__) || defined(__aarch64__)
        #define _arm64_
    #else
        #define _arm32_
    #endif
#endif

#if defined(_arm64_) || defined(_arm32_)
    #define _arm_
#endif

/* __ia64__ and __x86_64__      - defined by GNU C.
 * _M_IA64, _M_X64, _M_AMD64    - defined by Visual Studio.
 *
 * Microsoft can define _M_IX86, _M_AMD64 (before Visual Studio 8)
 * or _M_X64 (starting in Visual Studio 8).
 */
#if defined(__x86_64__) || defined(_M_X64)
    #define _x86_64_
#endif

#if defined(__i386__) || defined(_M_IX86)
    #define _i386_
#endif

#if defined(__ia64__) || defined(_M_IA64)
    #define _ia64_
#endif

#if defined(__powerpc__)
    #define _ppc_
#endif

#if defined(__powerpc64__)
    #define _ppc64_
#endif

#if defined(__wasm64__)
    #define _wasm64_
#endif

#if !defined(sparc) && !defined(__sparc) && !defined(__hpux__) && !defined(__alpha__) && !defined(_ia64_) && !defined(_x86_64_) && !defined(_arm_) && !defined(_i386_) && !defined(_ppc_) && !defined(_ppc64_) && !defined(_wasm64_)
    #error "platform not defined, please, define one"
#endif

#if defined(_x86_64_) || defined(_i386_)
    #define _x86_
#endif

#if defined(__MIC__)
    #define _mic_
    #define _k1om_
#endif

// stdio or MessageBox
#if defined(__CONSOLE__) || defined(_CONSOLE)
    #define _console_
#endif
#if (defined(_win_) && !defined(_console_))
    #define _windows_
#elif !defined(_console_)
    #define _console_
#endif

#if defined(__SSE__) || defined(SSE_ENABLED)
    #define _sse_
#endif

#if defined(__SSE2__) || defined(SSE2_ENABLED)
    #define _sse2_
#endif

#if defined(__SSE3__) || defined(SSE3_ENABLED)
    #define _sse3_
#endif

#if defined(__SSSE3__) || defined(SSSE3_ENABLED)
    #define _ssse3_
#endif

#if defined(__SSE4_1__) || defined(SSE41_ENABLED)
    #define _sse4_1_
#endif

#if defined(__SSE4_2__) || defined(SSE42_ENABLED)
    #define _sse4_2_
#endif

#if defined(__POPCNT__) || defined(POPCNT_ENABLED)
    #define _popcnt_
#endif

#if defined(__PCLMUL__) || defined(PCLMUL_ENABLED)
    #define _pclmul_
#endif

#if defined(__AES__) || defined(AES_ENABLED)
    #define _aes_
#endif

#if defined(__AVX__) || defined(AVX_ENABLED)
    #define _avx_
#endif

#if defined(__AVX2__) || defined(AVX2_ENABLED)
    #define _avx2_
#endif

#if defined(__FMA__) || defined(FMA_ENABLED)
    #define _fma_
#endif

#if defined(__DLL__) || defined(_DLL)
    #define _dll_
#endif

// 16, 32 or 64
#if defined(__sparc_v9__) || defined(_x86_64_) || defined(_ia64_) || defined(_arm64_) || defined(_ppc64_) || defined(_wasm64_)
    #define _64_
#else
    #define _32_
#endif

/* All modern 64-bit Unix systems use scheme LP64 (long, pointers are 64-bit).
 * Microsoft uses a different scheme: LLP64 (long long, pointers are 64-bit).
 *
 * Scheme          LP64   LLP64
 * char              8      8
 * short            16     16
 * int              32     32
 * long             64     32
 * long long        64     64
 * pointer          64     64
 */

#if defined(_32_)
    #define SIZEOF_PTR 4
#elif defined(_64_)
    #define SIZEOF_PTR 8
#endif

#define PLATFORM_DATA_ALIGN SIZEOF_PTR

#if !defined(SIZEOF_PTR)
    #error todo
#endif

#define SIZEOF_CHAR 1
#define SIZEOF_UNSIGNED_CHAR 1
#define SIZEOF_SHORT 2
#define SIZEOF_UNSIGNED_SHORT 2
#define SIZEOF_INT 4
#define SIZEOF_UNSIGNED_INT 4

#if defined(_32_)
    #define SIZEOF_LONG 4
    #define SIZEOF_UNSIGNED_LONG 4
#elif defined(_64_)
    #if defined(_win_)
        #define SIZEOF_LONG 4
        #define SIZEOF_UNSIGNED_LONG 4
    #else
        #define SIZEOF_LONG 8
        #define SIZEOF_UNSIGNED_LONG 8
    #endif // _win_
#endif // _32_

#if !defined(SIZEOF_LONG)
    #error todo
#endif

#define SIZEOF_LONG_LONG 8
#define SIZEOF_UNSIGNED_LONG_LONG 8

// clang-format on
