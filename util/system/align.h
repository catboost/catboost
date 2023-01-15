#pragma once

#include "yassert.h"
#include "defaults.h"
#include <util/generic/bitops.h>

template <class T>
static inline T AlignDown(T len, T align) noexcept {
    Y_ASSERT(IsPowerOf2(align)); // align should be power of 2
    return len & ~(align - 1);
}

template <class T>
static inline T AlignUp(T len, T align) noexcept {
    const T alignedResult = AlignDown(len + (align - 1), align);
    Y_ASSERT(alignedResult >= len); // check for overflow
    return alignedResult;
}

template <class T>
static inline T AlignUpSpace(T len, T align) noexcept {
    Y_ASSERT(IsPowerOf2(align));       // align should be power of 2
    return ((T)0 - len) & (align - 1); // AlignUp(len, align) - len;
}

template <class T>
static inline T* AlignUp(T* ptr, size_t align) noexcept {
    return (T*)AlignUp((uintptr_t)ptr, align);
}

template <class T>
static inline T* AlignDown(T* ptr, size_t align) noexcept {
    return (T*)AlignDown((uintptr_t)ptr, align);
}

template <class T>
static inline T AlignUp(T t) noexcept {
    return AlignUp(t, (size_t)PLATFORM_DATA_ALIGN);
}

template <class T>
static inline T AlignDown(T t) noexcept {
    return AlignDown(t, (size_t)PLATFORM_DATA_ALIGN);
}

template <class T>
static inline T Align(T t) noexcept {
    return AlignUp(t);
}
