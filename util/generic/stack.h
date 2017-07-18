#pragma once

#include "fwd.h"
#include "deque.h"

#include <stack>

template <class T, class S>
class ystack: public std::stack<T, S> {
    using TBase = std::stack<T, S>;

public:
    inline ystack()
        : TBase()
    {
    }

    explicit ystack(const S& s)
        : TBase(s)
    {
    }

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }
};
