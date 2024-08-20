#pragma once

#include "fwd.h"
#include "deque.h"
#include "vector.h"
#include "utility.h"

#include <util/str_stl.h>

#include <queue>

template <class T, class S>
class TQueue: public std::queue<T, S> {
    using TBase = std::queue<T, S>;

public:
    using TBase::TBase;

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }

    inline void clear() {
        this->c.clear();
    }

    inline S& Container() Y_LIFETIME_BOUND {
        return this->c;
    }

    inline const S& Container() const Y_LIFETIME_BOUND {
        return this->c;
    }
};

template <class T, class S, class C>
class TPriorityQueue: public std::priority_queue<T, S, C> {
    using TBase = std::priority_queue<T, S, C>;

public:
    using TBase::TBase;

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }

    inline void clear() {
        this->c.clear();
    }

    inline T PopValue() {
        Y_ASSERT(!this->empty());
        std::pop_heap(Container().begin(), Container().end(), this->comp);
        T value = std::move(Container().back());
        this->c.pop_back();
        return value;
    }

    inline S& Container() Y_LIFETIME_BOUND {
        return this->c;
    }

    inline const S& Container() const Y_LIFETIME_BOUND {
        return this->c;
    }
};
