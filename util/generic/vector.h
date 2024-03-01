#pragma once

#include "fwd.h"
#include "reserve.h"

#include <util/memory/alloc.h>

#include <vector>
#include <initializer_list>

#ifdef _YNDX_LIBCXX_ENABLE_VECTOR_POD_RESIZE_UNINITIALIZED
    #include <type_traits>
#endif

template <class T, class A>
class TVector: public std::vector<T, TReboundAllocator<A, T>> {
public:
    using TBase = std::vector<T, TReboundAllocator<A, T>>;
    using TSelf = TVector<T, A>;
    using size_type = typename TBase::size_type;

    inline TVector()
        : TBase()
    {
    }

    inline TVector(const typename TBase::allocator_type& a)
        : TBase(a)
    {
    }

    inline explicit TVector(::NDetail::TReserveTag rt)
        : TBase()
    {
        this->reserve(rt.Capacity);
    }

    inline explicit TVector(::NDetail::TReserveTag rt, const typename TBase::allocator_type& a)
        : TBase(a)
    {
        this->reserve(rt.Capacity);
    }

    inline explicit TVector(size_type count)
        : TBase(count)
    {
    }

    inline explicit TVector(size_type count, const typename TBase::allocator_type& a)
        : TBase(count, a)
    {
    }

    inline TVector(size_type count, const T& val)
        : TBase(count, val)
    {
    }

    inline TVector(size_type count, const T& val, const typename TBase::allocator_type& a)
        : TBase(count, val, a)
    {
    }

    inline TVector(std::initializer_list<T> il)
        : TBase(il)
    {
    }

    inline TVector(std::initializer_list<T> il, const typename TBase::allocator_type& a)
        : TBase(il, a)
    {
    }

    inline TVector(const TSelf& src)
        : TBase(src)
    {
    }

    inline TVector(TSelf&& src) noexcept
        : TBase(std::forward<TSelf>(src))
    {
    }

    template <class TIter>
    inline TVector(TIter first, TIter last)
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

    Y_PURE_FUNCTION inline bool empty() const noexcept {
        return TBase::empty();
    }

    inline yssize_t ysize() const noexcept {
        return (yssize_t)TBase::size();
    }

#if defined(_YNDX_LIBCXX_ENABLE_VECTOR_POD_RESIZE_UNINITIALIZED) && !defined(__CUDACC__)
    void yresize(size_type newSize) {
        if (std::is_standard_layout_v<T> && std::is_trivial_v<T>) {
            TBase::resize_uninitialized(newSize);
        } else {
            TBase::resize(newSize);
        }
    }
#else
    void yresize(size_type newSize) {
        TBase::resize(newSize);
    }
#endif

    inline void crop(size_type size) {
        if (this->size() > size) {
            this->erase(this->begin() + size, this->end());
        }
    }
};
