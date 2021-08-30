#pragma once

#include "fwd.h"

#include <util/memory/alloc.h>

#include <deque>
#include <memory>
#include <initializer_list>

template <class T, class A>
class TDeque: public std::deque<T, TReboundAllocator<A, T>> {
    using TBase = std::deque<T, TReboundAllocator<A, T>>;

public:
    using TBase::TBase;

    inline yssize_t ysize() const noexcept {
        return (yssize_t)this->size();
    }

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }
};
