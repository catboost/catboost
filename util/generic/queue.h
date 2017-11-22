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
    using TSelf = TQueue<T, S>;

public:
    inline TQueue() {
    }

    explicit TQueue(const S& ss)
        : TBase(ss)
    {
    }

    inline TQueue(const TSelf& src)
        : TBase(src)
    {
    }

    inline TQueue(TSelf&& src) noexcept
        : TBase(std::forward<TSelf>(src))
    {
    }

    inline TSelf& operator=(const TSelf& src) {
        TBase::operator=(src);
        return *this;
    }

    inline TSelf& operator=(TSelf&& src) noexcept {
        TBase::operator=(std::forward<TSelf>(src));
        return *this;
    }

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }

    inline void swap(TQueue& q) noexcept {
        this->c.swap(q.c);
    }

    inline void clear() {
        this->c.clear();
    }

    inline S& Container() {
        return this->c;
    }

    inline const S& Container() const {
        return this->c;
    }
};

template <class T, class S, class C>
class TPriorityQueue: public std::priority_queue<T, S, C> {
public:
    using TBase = std::priority_queue<T, S, C>;
    using TSelf = TPriorityQueue<T, S, C>;

    inline TPriorityQueue() {
    }

    explicit TPriorityQueue(const C& x)
        : TBase(x)
    {
    }

    inline TPriorityQueue(const C& x, const S& s)
        : TBase(x, s)
    {
    }

    inline TPriorityQueue(const C& x, S&& s)
        : TBase(x, std::move(s))
    {
    }

    template <class I>
    inline TPriorityQueue(I f, I l)
        : TBase(f, l)
    {
    }

    template <class I>
    inline TPriorityQueue(I f, I l, const C& x)
        : TBase(f, l, x)
    {
    }

    template <class I>
    inline TPriorityQueue(I f, I l, const C& x, const S& s)
        : TBase(f, l, x, s)
    {
    }

    inline TPriorityQueue(const TSelf& src)
        : TBase(src)
    {
    }

    inline TPriorityQueue(TSelf&& src) noexcept
        : TBase(std::forward<TSelf>(src))
    {
    }

    inline TSelf& operator=(const TSelf& src) {
        TBase::operator=(src);
        return *this;
    }

    inline TSelf& operator=(TSelf&& src) noexcept {
        TBase::operator=(std::forward<TSelf>(src));
        return *this;
    }

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }

    inline void clear() {
        this->c.clear();
    }

    inline void swap(TPriorityQueue& pq) {
        this->c.swap(pq.c);
        DoSwap(this->comp, pq.comp);
    }

    inline S& Container() {
        return this->c;
    }

    inline const S& Container() const {
        return this->c;
    }
};
