#pragma once

#include "fwd.h"

#include <util/memory/alloc.h>

#include <initializer_list>
#include <list>
#include <memory>
#include <utility>

template <class T, class A>
class TList: public std::list<T, TReboundAllocator<A, T>> {
    using TBase = std::list<T, TReboundAllocator<A, T>>;

public:
    using TBase::TBase;

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }
};
