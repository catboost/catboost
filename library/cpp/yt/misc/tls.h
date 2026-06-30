#pragma once

#include <util/system/compiler.h>

#define YT_PREVENT_TLS_CACHING Y_NO_INLINE

// Workaround for fiber (un)friendly TLS.
#define YT_DECLARE_THREAD_LOCAL(type, name) \
    type& name();

#define YT_DEFINE_THREAD_LOCAL(type, name, ...) \
    thread_local type name##Data { __VA_ARGS__ }; \
    Y_NO_INLINE type& name() \
    { \
        asm volatile(""); \
        return name##Data; \
    }

