#pragma once

#include "fwd.h"

#include <util/memory/alloc.h>

#include <initializer_list>
#include <list>
#include <memory>
#include <utility>

template <class T, class A>
class TList: public std::list<T, TReboundAllocator<A, T>> {
public:
    using TBase = std::list<T, TReboundAllocator<A, T>>;
    using TSelf = TList<T, A>;
    using allocator_type = typename TBase::allocator_type;
    using size_type = typename TBase::size_type;
    using value_type = typename TBase::value_type;

    inline TList() noexcept(std::is_nothrow_default_constructible<TBase>::value)
        : TBase()
    {
    }

    inline TList(size_type n, const T& val)
        : TBase(n, val)
    {
    }

    inline TList(const typename TBase::allocator_type& a)
        : TBase(a)
    {
    }

    template <typename InputIterator>
    inline TList(InputIterator first, InputIterator last)
        : TBase(first, last)
    {
    }

    inline TList(std::initializer_list<value_type> il, const allocator_type& alloc = allocator_type())
        : TBase(il, alloc)
    {
    }

    inline TList(const TSelf& src)
        : TBase(src)
    {
    }

    inline TList(TSelf&& src) noexcept
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

    Y_PURE_FUNCTION
    inline bool empty() const noexcept {
        return TBase::empty();
    }

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }
};
