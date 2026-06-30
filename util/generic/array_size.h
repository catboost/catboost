#pragma once

#include <cstddef>

namespace NArraySizePrivate {
    template <class T>
    struct TArraySize;

    template <class T, size_t N>
    struct TArraySize<T[N]> {
        enum {
            Result = N
        };
    };

    template <class T, size_t N>
    struct TArraySize<T (&)[N]> {
        enum {
            Result = N
        };
    };
} // namespace NArraySizePrivate

#define Y_ARRAY_SIZE(arr) ((size_t)::NArraySizePrivate::TArraySize<decltype(arr)>::Result)
