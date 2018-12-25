#pragma once

#include <util/generic/vector.h>
#include <util/ysaveload.h>

// A vector preallocated on the stack.
// After exceeding the preconfigured stack space falls back to the heap.
// Publicly inherits TVector, but disallows swap (and hence shrink_to_fit, also operator= is reimplemented via copying).
//
// Inspired by: http://qt-project.org/doc/qt-4.8/qvarlengtharray.html#details

template <typename T, size_t CountOnStack = 256, class Alloc = std::allocator<T>>
class TStackVec;

template <typename T, class Alloc = std::allocator<T>>
using TSmallVec = TStackVec<T, 16, Alloc>;

namespace NPrivate {
    template <class Alloc, class StackAlloc, typename T, typename U>
    struct TRebind {
        typedef TReboundAllocator<Alloc, U> other;
    };

    template <class Alloc, class StackAlloc, typename T>
    struct TRebind<Alloc, StackAlloc, T, T> {
        typedef StackAlloc other;
    };

    template <typename T, size_t CountOnStack, class Alloc = std::allocator<T>>
    class TStackBasedAllocator: public Alloc {
    public:
        typedef TStackBasedAllocator<T, CountOnStack, Alloc> TSelf;

        using typename Alloc::const_pointer;
        using typename Alloc::const_reference;
        using typename Alloc::difference_type;
        using typename Alloc::pointer;
        using typename Alloc::reference;
        using typename Alloc::size_type;
        using typename Alloc::value_type;

        template <class U>
        struct rebind: public ::NPrivate::TRebind<Alloc, TSelf, T, U> {
        };

    public:
        TStackBasedAllocator()
            : StorageUsed(false)
        {
        }

        pointer allocate(size_type n, std::allocator<void>::const_pointer hint = nullptr) {
            if (!StorageUsed && n <= CountOnStack) {
                StorageUsed = true;
                return reinterpret_cast<pointer>(StackBasedStorage);
            } else {
                return FallbackAllocator.allocate(n, hint);
            }
        }

        void deallocate(pointer p, size_type n) {
            if (p == reinterpret_cast<pointer>(StackBasedStorage)) {
                StorageUsed = false;
            } else {
                FallbackAllocator.deallocate(p, n);
            }
        }

        using Alloc::address;
        using Alloc::construct;
        using Alloc::destroy;
        using Alloc::max_size;

    private:
        char StackBasedStorage[CountOnStack * sizeof(T)];
        bool StorageUsed;
        Alloc FallbackAllocator;
    };
}

template <typename T, size_t CountOnStack, class Alloc>
class TStackVec: public TVector<T, ::NPrivate::TStackBasedAllocator<T, CountOnStack, TReboundAllocator<Alloc, T>>> {
public:
    typedef TVector<T, ::NPrivate::TStackBasedAllocator<T, CountOnStack, TReboundAllocator<Alloc, T>>> TBase;
    typedef TStackVec<T, CountOnStack, TReboundAllocator<Alloc, T>> TSelf;

    using typename TBase::const_iterator;
    using typename TBase::const_reverse_iterator;
    using typename TBase::iterator;
    using typename TBase::reverse_iterator;
    using typename TBase::size_type;
    using typename TBase::value_type;

public:
    inline TStackVec()
        : TBase()
    {
        TBase::reserve(CountOnStack);
    }

    inline explicit TStackVec(size_type count)
        : TBase()
    {
        if (count <= CountOnStack) {
            TBase::reserve(CountOnStack);
        }
        TBase::resize(count);
    }

    inline TStackVec(size_type count, const T& val)
        : TBase()
    {
        if (count <= CountOnStack) {
            TBase::reserve(CountOnStack);
        }
        TBase::assign(count, val);
    }

    inline TStackVec(const TSelf& src)
        : TBase(src.begin(), src.end())
    {
    }

    template <class A>
    inline TStackVec(const TVector<T, A>& src)
        : TBase(src.begin(), src.end())
    {
    }

    inline TStackVec(std::initializer_list<T> il)
        : TBase(il.begin(), il.end())
    {
    }

    template <class TIter>
    inline TStackVec(TIter first, TIter last)
        : TBase(first, last)
    {
    }

public:
    void swap(TSelf&) = delete;
    void shrink_to_fit() = delete;

    inline TSelf& operator=(const TSelf& src) {
        TBase::assign(src.begin(), src.end());
        return *this;
    }

    template <class A>
    inline TSelf& operator=(const TVector<T, A>& src) {
        TBase::assign(src.begin(), src.end());
        return *this;
    }

    inline TSelf& operator=(std::initializer_list<T> il) {
        TBase::assign(il.begin(), il.end());
        return *this;
    }
};

template <typename T, size_t CountOnStack, class Alloc>
class TSerializer<TStackVec<T, CountOnStack, Alloc>>: public TVectorSerializer<TStackVec<T, CountOnStack, Alloc>> {
};
