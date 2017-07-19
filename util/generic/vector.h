#pragma once

#include "fwd.h"
#include "reserve.h"

#include <util/memory/alloc.h>

#include <vector>
#include <initializer_list>

template <class T, class A>
class yvector: public std::vector<T, TReboundAllocator<A, T>> {
public:
    using TBase = std::vector<T, TReboundAllocator<A, T>>;
    using TSelf = yvector<T, A>;
    using size_type = typename TBase::size_type;

    inline yvector()
        : TBase()
    {
    }

    inline yvector(const typename TBase::allocator_type& a)
        : TBase(a)
    {
    }

    inline explicit yvector(::NDetail::TReserveTag rt)
        : TBase()
    {
        this->reserve(rt.Capacity);
    }

    inline explicit yvector(::NDetail::TReserveTag rt, const typename TBase::allocator_type& a)
        : TBase(a)
    {
        this->reserve(rt.Capacity);
    }

    inline explicit yvector(size_type count)
        : TBase(count)
    {
    }

    inline yvector(size_type count, const T& val)
        : TBase(count, val)
    {
    }

    inline yvector(size_type count, const T& val, const typename TBase::allocator_type& a)
        : TBase(count, val, a)
    {
    }

    inline yvector(std::initializer_list<T> il)
        : TBase(il)
    {
    }

    inline yvector(std::initializer_list<T> il, const typename TBase::allocator_type& a)
        : TBase(il, a)
    {
    }

    inline yvector(const TSelf& src)
        : TBase(src)
    {
    }

    inline yvector(TSelf&& src) noexcept
        : TBase(std::forward<TSelf>(src))
    {
    }

    template <class TIter>
    inline yvector(TIter first, TIter last)
        : TBase(first, last)
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

    inline TSelf& operator=(std::initializer_list<T> il) {
        this->assign(il.begin(), il.end());
        return *this;
    }

    inline explicit operator bool() const noexcept {
        return !this->empty();
    }

    inline size_type operator+() const noexcept {
        return this->size();
    }

    inline T* operator~() noexcept {
        return this->data();
    }

    inline const T* operator~() const noexcept {
        return this->data();
    }

    inline yssize_t ysize() const noexcept {
        return (yssize_t)TBase::size();
    }

    inline void crop(size_type size) {
        if (this->size() > size) {
            this->resize(size);
        }
    }
};
