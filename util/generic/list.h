#pragma once

#include "fwd.h"

#include <util/memory/alloc.h>

#include <initializer_list>
#include <list>
#include <memory>
#include <utility>

template <class T, class A>
class ylist: public std::list<T, TReboundAllocator<A, T>> {
public:
    using TBase = std::list<T, TReboundAllocator<A, T>>;
    using TSelf = ylist<T, A>;
    using allocator_type = typename TBase::allocator_type;
    using size_type = typename TBase::size_type;
    using value_type = typename TBase::value_type;

    inline ylist()
        : TBase()
    {
    }

    inline ylist(size_type n, const T& val)
        : TBase(n, val)
    {
    }

    inline ylist(const typename TBase::allocator_type& a)
        : TBase(a)
    {
    }

    template <typename InputIterator>
    inline ylist(InputIterator first, InputIterator last)
        : TBase(first, last)
    {
    }

    inline ylist(std::initializer_list<value_type> il, const allocator_type& alloc = allocator_type())
        : TBase(il, alloc)
    {
    }

    inline ylist(const TSelf& src)
        : TBase(src)
    {
    }

    inline ylist(TSelf&& src) noexcept
        : TBase(std::forward<TBase>(src))
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
};
