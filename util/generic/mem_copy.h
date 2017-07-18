#pragma once

#include "typetraits.h"

#include <util/system/types.h>

#include <cstring>

template <class T>
using TIfPOD = std::enable_if_t<TTypeTraits<T>::IsPod, T*>;

template <class T>
using TIfNotPOD = std::enable_if_t<!TTypeTraits<T>::IsPod, T*>;

template <class T>
static inline TIfPOD<T> MemCopy(T* to, const T* from, size_t n) noexcept {
    if (n) {
        memcpy(to, from, n * sizeof(T));
    }

    return to;
}

template <class T>
static inline TIfNotPOD<T> MemCopy(T* to, const T* from, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        to[i] = from[i];
    }

    return to;
}

template <class T>
static inline TIfPOD<T> MemMove(T* to, const T* from, size_t n) noexcept {
    if (n) {
        memmove(to, from, n * sizeof(T));
    }

    return to;
}

template <class T>
static inline TIfNotPOD<T> MemMove(T* to, const T* from, size_t n) {
    if (to <= from || to >= from + n) {
        return MemCopy(to, from, n);
    }

    //copy backwards
    while (n) {
        to[n - 1] = from[n - 1];
        --n;
    }

    return to;
}
