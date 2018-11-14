#pragma once

#include "defaults.h"
#include "src_root.h"
#include "backtrace.h"

#if defined(_MSC_VER)
#include <new>
#if defined(_DEBUG)
#if defined(_CRTDBG_MAP_ALLOC)
#include <cstdlib>  /* definitions for malloc/calloc */
#include <malloc.h> /* must be before their redefinitions as _*_dbg() */
#endif
#include <crtdbg.h>
#else
#endif
#include <cassert>
#elif defined(__GNUC__)
#ifdef _sun_
#include <alloca.h>
#endif
#include <cassert>
#endif

#if !defined(_MSC_VER)

#if defined(__x86_64__) || defined(__i386__)
#define __debugbreak ydebugbreak
inline void ydebugbreak() {
    __asm__ volatile("int $3\n");
}
#else
inline void __debugbreak() {
    assert(0);
}
#endif

inline bool YaIsDebuggerPresent() {
    return false;
}

#else
// __debugbreak is intrinsic in MSVC

extern "C" {
__declspec(dllimport) int __stdcall IsDebuggerPresent();
}

inline bool YaIsDebuggerPresent() {
    return IsDebuggerPresent() != 0;
}
#endif

#undef Y_ASSERT

#if !defined(NDEBUG) && !defined(__GCCXML__)
#define Y_ASSERT(a)                                         \
    do {                                                    \
        try {                                               \
            if (Y_UNLIKELY(!(a))) {                         \
                if (YaIsDebuggerPresent())                  \
                    __debugbreak();                         \
                else {                                      \
                    PrintBackTrace();                       \
                    assert(false && (a));                   \
                }                                           \
            }                                               \
        } catch (...) {                                     \
            if (YaIsDebuggerPresent())                      \
                __debugbreak();                             \
            else {                                          \
                PrintBackTrace();                           \
                assert(false && "Exception during assert"); \
            }                                               \
        }                                                   \
    } while (false)
#else
#define Y_ASSERT(a)                            \
    do {                                       \
        if (false) {                           \
            auto __xxx = static_cast<bool>(a); \
            Y_UNUSED(__xxx);                   \
        }                                      \
    } while (false)
#endif

namespace NPrivate {
    /// method should not be used directly
    [[noreturn]] void Panic(const TStaticBuf& file, int line, const char* function, const char* expr, const char* format, ...) noexcept Y_PRINTF_FORMAT(5, 6);
}

/// Assert that does not depend on NDEBUG macro and outputs message like printf
#define Y_VERIFY(expr, ...)                                                                          \
    do {                                                                                             \
        if (Y_UNLIKELY(!(expr))) {                                                                   \
            ::NPrivate::Panic(__SOURCE_FILE_IMPL__, __LINE__, __FUNCTION__, #expr, " " __VA_ARGS__); \
        }                                                                                            \
    } while (false)

#define Y_FAIL(...)                                                                                \
    do {                                                                                           \
        ::NPrivate::Panic(__SOURCE_FILE_IMPL__, __LINE__, __FUNCTION__, nullptr, " " __VA_ARGS__); \
    } while (false)

#ifndef NDEBUG
/// Assert that depend on NDEBUG macro and outputs message like printf
#define Y_VERIFY_DEBUG(expr, ...)                                                                    \
    do {                                                                                             \
        if (Y_UNLIKELY(!(expr))) {                                                                   \
            ::NPrivate::Panic(__SOURCE_FILE_IMPL__, __LINE__, __FUNCTION__, #expr, " " __VA_ARGS__); \
        }                                                                                            \
    } while (false)
#else
#define Y_VERIFY_DEBUG(expr, ...)                 \
    do {                                          \
        if (false) {                              \
            bool __xxx = static_cast<bool>(expr); \
            Y_UNUSED(__xxx);                      \
        }                                         \
    } while (false)
#endif
