#pragma once

#if defined(WITH_VALGRIND) && defined(HAVE_VALGRIND)
    #include <valgrind/valgrind.h>
    #include <valgrind/memcheck.h>

    #if !defined(VALGRIND_CHECK_READABLE)
        #define VALGRIND_CHECK_READABLE(s, l) VALGRIND_CHECK_MEM_IS_DEFINED(s, l)
    #endif

    #if !defined(VALGRIND_MAKE_READABLE)
        #define VALGRIND_MAKE_READABLE(a, b) VALGRIND_MAKE_MEM_DEFINED(a, b)
    #endif
#else
    #define RUNNING_ON_VALGRIND 0
    #define VALGRIND_CHECK_READABLE(s, l)
    #define VALGRIND_MAKE_READABLE(a, b) 0
    #define VALGRIND_STACK_REGISTER(start, end) 0
    #define VALGRIND_STACK_DEREGISTER(id)
    #define VALGRIND_DISCARD(v) ((void)v)
static inline int VALGRIND_PRINTF(...) {
    return 0;
}
    #define VALGRIND_DO_LEAK_CHECK
#endif

namespace NValgrind {
    inline constexpr static bool ValgrindIsOn() noexcept {
#if defined(WITH_VALGRIND)
        return true;
#else
        return false;
#endif
    }

    // Returns valgrinded if running under Valgrind and plain otherwise
    // Ment to be used in test code for constants (timeouts, etc)
    template <typename T>
    inline constexpr static T PlainOrUnderValgrind(T plain, T valgrinded) noexcept {
#if defined(WITH_VALGRIND)
        Y_UNUSED(plain);
        return valgrinded;
#else
        Y_UNUSED(valgrinded);
        return plain;
#endif
    }
}
