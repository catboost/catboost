#pragma once

#include "range_ops.h"

#include <util/generic/array_ref.h>
#include <util/system/defaults.h> /* For alloca. */

namespace NStackArray {
    /**
     * A stack-allocated array. Should be used instead of � variable length
     * arrays that are not part of C++ standard.
     *
     * Example usage:
     * @code
     * void f(int size) {
     *    // T array[size]; // Wrong!
     *    TStackArray<T> array(ALLOC_ON_STACK(T, size)); // Right!
     *    // ...
     * }
     * @endcode
     *
     * Note that it is generally a *VERY BAD* idea to use this in inline methods
     * as those might be called from a loop, and then stack overflow is in the cards.
     */
    template <class T>
    class TStackArray: public TArrayRef<T> {
    public:
        inline TStackArray(void* data, size_t len)
            : TArrayRef<T>((T*)data, len)
        {
            NRangeOps::InitializeRange(this->begin(), this->end());
        }

        inline ~TStackArray() {
            NRangeOps::DestroyRange(this->begin(), this->end());
        }
    };
}

#define ALLOC_ON_STACK(type, n) alloca(sizeof(type) * (n)), (n)
