#pragma once

#include <util/generic/vector.h>
#include <util/ysaveload.h>

#include <type_traits>

// A vector preallocated on the stack.
// After exceeding the preconfigured stack space falls back to the heap.
// Publicly inherits TVector, but disallows swap (and hence shrink_to_fit, also operator= is reimplemented via copying).
//
// Inspired by: http://qt-project.org/doc/qt-4.8/qvarlengtharray.html#details

template <typename T, size_t CountOnStack = 256, bool UseFallbackAlloc = true, class Alloc = std::allocator<T>>
class TStackVec;

template <typename T, class Alloc = std::allocator<T>>
using TSmallVec = TStackVec<T, 16, true, Alloc>;

namespace NPrivate {
    template <class Alloc, class StackAlloc, typename T, typename U>
    struct TRebind {
        typedef TReboundAllocator<Alloc, U> other;
    };

    template <class Alloc, class StackAlloc, typename T>
    struct TRebind<Alloc, StackAlloc, T, T> {
        typedef StackAlloc other;
    };

    template <typename T, size_t CountOnStack, bool UseFallbackAlloc, class Alloc = std::allocator<T>>
    class TStackBasedAllocator: public Alloc {
    public:
        typedef TStackBasedAllocator<T, CountOnStack, UseFallbackAlloc, Alloc> TSelf;

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
        pointer allocate(size_type n, std::allocator<void>::const_pointer hint = nullptr) {
            if (CountOnStack >= n + StorageSpent) {
                const auto result = reinterpret_cast<pointer>(&StackBasedStorage[StorageSpent]);
                StorageSpent += n;
                ++StackAllocations;
                return result;
            } else {
                if constexpr (!UseFallbackAlloc) {
                    Y_FAIL("Stack storage overflow");
                }
                return FallbackAllocator.allocate(n, hint);
            }
        }


        void deallocate(pointer p, size_type n) {
            if (p >= reinterpret_cast<pointer>(&StackBasedStorage[0]) &&
                    p < reinterpret_cast<pointer>(&StackBasedStorage[CountOnStack])) {
                --StackAllocations;
                if (!StackAllocations) {
                    StorageSpent = 0;
                }
            } else {
                FallbackAllocator.deallocate(p, n);
            }
        }

        using Alloc::address;
        using Alloc::construct;
        using Alloc::destroy;
        using Alloc::max_size;

    private:
        std::aligned_storage_t<sizeof(T), alignof(T)> StackBasedStorage[CountOnStack];
        // Number of cells (out of CountOnStack) at the beginning of the storage that may be occupied.
        size_t StorageSpent = 0;
        // How many blocks of memory are allocated from our storage.
        // When StackAllocations > 0, only cells with indices greater than StorageSpent are guaranteed to be unused.
        // When StackAllocations == 0, all cells are unused.
        size_t StackAllocations = 0;
        Alloc FallbackAllocator;
    };
}

template <typename T, size_t CountOnStack, bool UseFallbackAlloc, class Alloc>
class TStackVec: public TVector<T, ::NPrivate::TStackBasedAllocator<T, CountOnStack, UseFallbackAlloc, TReboundAllocator<Alloc, T>>> {
public:
    typedef TVector<T, ::NPrivate::TStackBasedAllocator<T, CountOnStack, UseFallbackAlloc, TReboundAllocator<Alloc, T>>> TBase;
    typedef TStackVec<T, CountOnStack, UseFallbackAlloc, TReboundAllocator<Alloc, T>> TSelf;

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

    // NB(eeight) The following four constructors all have a drawback -- we cannot pre-reserve our storage.
    // This leads to suboptimal stack storage utilization.
    // For example, if we copy a TStackVec of size one, the new TStackVec would initially have capacity
    // of only one. After that it is not possible to reallocate to full stack storage because of how c++
    // allocators work.
    // Our allocator will try its best and fit successfull allocations into the same stack buffer which
    // (if CountOnStack is 16) will allow the vector to grow to up to 8 elemets without spilling over
    // to allocated memory. It looks like this is the best we can do under the circumstances.
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
class TSerializer<TStackVec<T, CountOnStack, true, Alloc>>: public TVectorSerializer<TStackVec<T, CountOnStack, true, Alloc>> {
};

template <typename T, size_t CountOnStack, class Alloc>
class TSerializer<TStackVec<T, CountOnStack, false, Alloc>> {
public:
    static void Save(IOutputStream* rh, const TStackVec<T, CountOnStack, false, Alloc>& v) {
        if constexpr (CountOnStack < 256) {
            ::Save(rh, (ui8)v.size());
        } else {
            ::Save(rh, v.size());
        }
        ::SaveArray(rh, v.data(), v.size());
    }

    static void Load(IInputStream* rh, TStackVec<T, CountOnStack, false, Alloc>& v) {
        std::conditional_t<CountOnStack < 256, ui8, size_t> size;
        ::Load(rh, size);
        v.resize(size);
        ::LoadPodArray(rh, v.data(), v.size());
    }
};
