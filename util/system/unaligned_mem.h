#pragma once

#include "defaults.h"

#include <string.h>

// The following code used to have smart tricks assuming that unaligned reads and writes are OK on x86. This assumption
// is wrong because compiler may emit alignment-sensitive x86 instructions e.g. movaps. See IGNIETFERRO-735.

template <class T>
inline T ReadUnaligned(const void* from) noexcept {
    T ret;
    memcpy(&ret, from, sizeof(T));
    return ret;
}

template <class T>
inline void WriteUnaligned(void* to, const T& t) noexcept {
    memcpy(to, &t, sizeof(T));
}
