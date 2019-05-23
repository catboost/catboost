#pragma once

#include "fwd.h"
#include "deque.h"

#include <stack>

template <class T, class S>
class TStack: public std::stack<T, S> {
    using TBase = std::stack<T, S>;

public:
    using TBase::TBase;

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }
};
