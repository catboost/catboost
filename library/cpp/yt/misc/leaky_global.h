#pragma once

#include <new>

//! Defines a global variable that is initialized on its first access and is never destroyed.
/*!
 *  In contrast to a usual variable with static storage duration, this one is not
 *  susceptible to initialization and destruction order fiasco issues: it is
 *  constructed in a static buffer, and such globals (e.g. loggers) may be
 *  legitimately accessed during static destruction.
 */
#if defined(_asan_enabled_) || defined(_lsan_enabled_)

#include <sanitizer/lsan_interface.h>

// Whatever the constructor allocates is never freed, so keep it out of leak reports.
#define YT_DEFINE_LEAKY_GLOBAL(type, name, ...) \
    inline type& name() \
    { \
        alignas(type) static char Storage[sizeof(type)]; \
        static type* Result = [] { \
            __lsan_disable(); \
            auto* result = new (&Storage) type{__VA_ARGS__}; \
            __lsan_enable(); \
            return result; \
        }(); \
        return *Result; \
    }

#else

#define YT_DEFINE_LEAKY_GLOBAL(type, name, ...) \
    inline type& name() \
    { \
        alignas(type) static char Storage[sizeof(type)]; \
        static type* Result = new (&Storage) type{__VA_ARGS__}; \
        return *Result; \
    }

#endif
