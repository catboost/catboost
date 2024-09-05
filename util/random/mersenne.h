#pragma once

#include "mersenne64.h"
#include "mersenne32.h"
#include "common_ops.h"

namespace NPrivate {
    template <class T>
    struct TMersenneTraits;

    template <>
    struct TMersenneTraits<ui64> {
        using TImpl = TMersenne64;
    };

    template <>
    struct TMersenneTraits<ui32> {
        using TImpl = TMersenne32;
    };
} // namespace NPrivate

class IInputStream;

template <class T>
class TMersenne: public TCommonRNG<T, TMersenne<T>>, public ::NPrivate::TMersenneTraits<T>::TImpl {
public:
    using TBase = typename ::NPrivate::TMersenneTraits<T>::TImpl;

    inline TMersenne() noexcept {
    }

    inline TMersenne(T seed) noexcept
        : TBase(seed)
    {
    }

    inline TMersenne(IInputStream& pool)
        : TBase(pool)
    {
    }

    inline TMersenne(const T* keys, size_t len) noexcept
        : TBase(keys, len)
    {
    }
};
