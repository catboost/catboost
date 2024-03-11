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
    #if defined(__has_builtin) && __has_builtin(__debugbreak)
    // Do nothing, use __debugbreak builtin
    #elif defined(__has_builtin) && __has_builtin(__builtin_debugtrap)
inline void __debugbreak() {
    __builtin_debugtrap();
}
    #else
inline void __debugbreak() {
        #if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("int $3\n");
        #else
    assert(0);
        #endif
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

inline void YaDebugBreak() {
    __debugbreak();
}

#undef Y_ASSERT

#if !defined(NDEBUG) && !defined(__GCCXML__)
    #define Y_HIT_DEBUGGER()             \
        do {                             \
            if (YaIsDebuggerPresent()) { \
                __debugbreak();          \
            }                            \
        } while (false)
#else
    #define Y_HIT_DEBUGGER() Y_SEMICOLON_GUARD
#endif

namespace NPrivate {
    /// method should not be used directly
    [[noreturn]] void Panic(const TStaticBuf& file, int line, const char* function, const char* expr, const char* format, ...) noexcept Y_PRINTF_FORMAT(5, 6);
}

/// Assert that does not depend on NDEBUG macro and outputs message like printf
#define Y_ABORT_UNLESS(expr, ...)                                                                    \
    do {                                                                                             \
        if (Y_UNLIKELY(!(expr))) {                                                                   \
            Y_HIT_DEBUGGER();                                                                        \
            /* NOLINTNEXTLINE */                                                                     \
            ::NPrivate::Panic(__SOURCE_FILE_IMPL__, __LINE__, __FUNCTION__, #expr, " " __VA_ARGS__); \
        }                                                                                            \
    } while (false)

#define Y_ABORT_IF(expr, ...) Y_ABORT_UNLESS(!(expr), __VA_ARGS__)
#define Y_ABORT(...) Y_ABORT_UNLESS(false, __VA_ARGS__)

#ifndef NDEBUG
    /// Assert that depend on NDEBUG macro and outputs message like printf
    #define Y_DEBUG_ABORT_UNLESS Y_ABORT_UNLESS
#else
    #define Y_DEBUG_ABORT_UNLESS(expr, ...)           \
        do {                                          \
            if (false) {                              \
                bool __xxx = static_cast<bool>(expr); \
                Y_UNUSED(__xxx);                      \
            }                                         \
        } while (false)
#endif
#define Y_ASSERT(a) Y_DEBUG_ABORT_UNLESS(a)
