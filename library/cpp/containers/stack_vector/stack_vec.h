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

template <typename T, size_t CountOnStack = 256>
using TStackOnlyVec = TStackVec<T, CountOnStack, false>;

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
            if (!IsStorageUsed && CountOnStack >= n) {
                IsStorageUsed = true;
                return reinterpret_cast<pointer>(&StackBasedStorage[0]);
            } else {
                if constexpr (!UseFallbackAlloc) {
                    Y_FAIL(
                            "Stack storage overflow. Capacity: %d, requested: %d", (int)CountOnStack, int(n));
                }
                return FallbackAllocator().allocate(n, hint);
            }
        }


        void deallocate(pointer p, size_type n) {
            if (p >= reinterpret_cast<pointer>(&StackBasedStorage[0]) &&
                    p < reinterpret_cast<pointer>(&StackBasedStorage[CountOnStack])) {
                Y_VERIFY(IsStorageUsed);
                IsStorageUsed = false;
            } else {
                FallbackAllocator().deallocate(p, n);
            }
        }

    private:
        std::aligned_storage_t<sizeof(T), alignof(T)> StackBasedStorage[CountOnStack];
        bool IsStorageUsed = false;

    private:
        Alloc& FallbackAllocator() noexcept {
            return static_cast<Alloc&>(*this);
        }
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
    TStackVec()
        : TBase()
    {
        TBase::reserve(CountOnStack);
    }

    explicit TStackVec(size_type count)
        : TBase()
    {
        if (count <= CountOnStack) {
            TBase::reserve(CountOnStack);
        }
        TBase::resize(count);
    }

    TStackVec(size_type count, const T& val)
        : TBase()
    {
        if (count <= CountOnStack) {
            TBase::reserve(CountOnStack);
        }
        TBase::assign(count, val);
    }

    TStackVec(const TSelf& src)
        : TStackVec(src.begin(), src.end())
    {
    }

    template <class A>
    TStackVec(const TVector<T, A>& src)
        : TStackVec(src.begin(), src.end())
    {
    }

    TStackVec(std::initializer_list<T> il)
        : TStackVec(il.begin(), il.end())
    {
    }

    template <class TIter>
    TStackVec(TIter first, TIter last)
    {
        // NB(eeight) Since we want to call 'reserve' here, we cannot just delegate to TVector ctor.
        // The best way to insert values afterwards is to call TVector::insert. However there is a caveat.
        // In order to call this ctor of TVector, T needs to be just move-constructible. Insert however
        // requires T to be move-assignable.
        TBase::reserve(CountOnStack);
        if constexpr (std::is_move_assignable_v<T>) {
            // Fast path
            TBase::insert(TBase::end(), first, last);
        } else {
            // Slow path.
            for (; first != last; ++first) {
                TBase::push_back(*first);
            }
        }
    }

public:
    void swap(TSelf&) = delete;
    void shrink_to_fit() = delete;

    TSelf& operator=(const TSelf& src) {
        TBase::assign(src.begin(), src.end());
        return *this;
    }

    template <class A>
    TSelf& operator=(const TVector<T, A>& src) {
        TBase::assign(src.begin(), src.end());
        return *this;
    }

    TSelf& operator=(std::initializer_list<T> il) {
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
