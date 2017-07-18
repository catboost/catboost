#pragma once

#include "defaults.h"

#if defined(_x86_64_) || defined(_i386_) || defined(_arm64_)
#define UNALIGNED_ACCESS_OK
#endif

#if !defined(UNALIGNED_ACCESS_OK)
#include <string.h>
#endif

template <class T>
inline T ReadUnaligned(const void* from) noexcept Y_NO_SANITIZE("undefined") {
#if defined(UNALIGNED_ACCESS_OK)
    using TAliasedType = T alias_hack;

    static_assert(sizeof(T) == sizeof(TAliasedType), "ups, some completely wrong here");

    return *(TAliasedType*)from;
#else
    T ret;

    memcpy(&ret, from, sizeof(T));

    return ret;
#endif
}

template <class T>
inline void WriteUnaligned(void* to, const T& t) noexcept Y_NO_SANITIZE("undefined") {
#if defined(UNALIGNED_ACCESS_OK)
    using TAliasedType = T alias_hack;

    static_assert(sizeof(T) == sizeof(TAliasedType), "ups, some completely wrong here");

    *(TAliasedType*)to = t;
#else
    memcpy(to, &t, sizeof(T));
#endif
}
