#pragma once

#include "ref_counted.h"

#include <util/generic/hash.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
class TWeakPtr
{
public:
    using TUnderlying = T;
    using element_type = T;

    //! Empty constructor.
    TWeakPtr() = default;

    TWeakPtr(std::nullptr_t)
    { }

    //! Constructor from an unqualified reference.
    /*!
     * Note that this constructor could be racy due to unsynchronized operations
     * on the object and on the counter.
     */
    explicit TWeakPtr(T* p) noexcept
        : T_(p)
    {

#if defined(_tsan_enabled_)
        if (T_) {
            RefCounter_ = GetRefCounter(T_);
        }
#endif
        AcquireRef();
    }

    //! Constructor from a strong reference.
    TWeakPtr(const TIntrusivePtr<T>& ptr) noexcept
        : TWeakPtr(ptr.Get())
    { }

    //! Constructor from a strong reference with an upcast.
    template <class U, class = typename std::enable_if_t<std::is_convertible_v<U*, T*>>>
    TWeakPtr(const TIntrusivePtr<U>& ptr) noexcept
        : TWeakPtr(ptr.Get())
    {
        static_assert(
            std::derived_from<T, TRefCountedBase>,
            "Cast allowed only for types derived from TRefCountedBase");
    }

    //! Copy constructor.
    TWeakPtr(const TWeakPtr& other) noexcept
        : TWeakPtr(other.T_)
    { }

    //! Copy constructor with an upcast.
    template <class U, class = typename std::enable_if_t<std::is_convertible_v<U*, T*>>>
    TWeakPtr(const TWeakPtr<U>& other) noexcept
        : TWeakPtr(other.Lock())
    {
        static_assert(
            std::derived_from<T, TRefCountedBase>,
            "Cast allowed only for types derived from TRefCountedBase");
    }

    //! Move constructor.
    TWeakPtr(TWeakPtr&& other) noexcept
    {
        other.Swap(*this);
    }

    //! Move constructor with an upcast.
    template <class U, class = typename std::enable_if_t<std::is_convertible_v<U*, T*>>>
    TWeakPtr(TWeakPtr<U>&& other) noexcept
    {
        static_assert(
            std::derived_from<T, TRefCountedBase>,
            "Cast allowed only for types derived from TRefCountedBase");
        TIntrusivePtr<U> strongOther = other.Lock();
        if (strongOther) {
            T_ = other.T_;
            other.T_ = nullptr;

#if defined(_tsan_enabled_)
            RefCounter_ = other.RefCounter_;
            other.RefCounter_ = nullptr;
#endif
        }
    }

    //! Destructor.
    ~TWeakPtr()
    {
        ReleaseRef();
    }

    //! Assignment operator from a strong reference.
    template <class U>
    TWeakPtr& operator=(const TIntrusivePtr<U>& ptr) noexcept
    {
        static_assert(
            std::is_convertible_v<U*, T*>,
            "U* must be convertible to T*");
        TWeakPtr(ptr).Swap(*this);
        return *this;
    }

    //! Copy assignment operator.
    TWeakPtr& operator=(const TWeakPtr& other) noexcept
    {
        TWeakPtr(other).Swap(*this);
        return *this;
    }

    //! Copy assignment operator with an upcast.
    template <class U>
    TWeakPtr& operator=(const TWeakPtr<U>& other) noexcept
    {
        static_assert(
            std::is_convertible_v<U*, T*>,
            "U* must be convertible to T*");
        TWeakPtr(other).Swap(*this);
        return *this;
    }

    //! Move assignment operator.
    TWeakPtr& operator=(TWeakPtr&& other) noexcept
    {
        other.Swap(*this);
        return *this;
    }

    //! Move assignment operator with an upcast.
    template <class U>
    TWeakPtr& operator=(TWeakPtr<U>&& other) noexcept
    {
        static_assert(
            std::is_convertible_v<U*, T*>,
            "U* must be convertible to T*");
        TWeakPtr(std::move(other)).Swap(*this);
        return *this;
    }

    //! Drop the pointer.
    void Reset() // noexcept
    {
        TWeakPtr().Swap(*this);
    }

    //! Replace the pointer with a specified one.
    void Reset(T* p) // noexcept
    {
        TWeakPtr(p).Swap(*this);
    }

    //! Replace the pointer with a specified one.
    template <class U>
    void Reset(const TIntrusivePtr<U>& ptr) // noexcept
    {
        static_assert(
            std::is_convertible_v<U*, T*>,
            "U* must be convertible to T*");
        TWeakPtr(ptr).Swap(*this);
    }

    //! Swap the pointer with the other one.
    void Swap(TWeakPtr& other) noexcept
    {
        DoSwap(T_, other.T_);
#if defined(_tsan_enabled_)
        DoSwap(RefCounter_, other.RefCounter_);
#endif
    }

    //! Acquire a strong reference to the pointee and return a strong pointer.
    TIntrusivePtr<T> Lock() const noexcept
    {
        return T_ && RefCounter()->TryRef()
            ? TIntrusivePtr<T>(T_, false)
            : TIntrusivePtr<T>();
    }

    bool IsExpired() const noexcept
    {
        return !T_ || (RefCounter()->GetRefCount() == 0);
    }

    const TRefCounter* TryGetRefCounter() const
    {
        return T_
            ? RefCounter()
            : nullptr;
    }

private:
    void AcquireRef()
    {
        if (T_) {
            RefCounter()->WeakRef();
        }
    }

    void ReleaseRef()
    {
        if (T_) {
            // Support incomplete type.
            if (RefCounter()->WeakUnref()) {
                DeallocateRefCounted(T_);
            }
        }
    }

    template <class U>
    friend class TWeakPtr;
    template <class U>
    friend struct ::THash;

    T* T_ = nullptr;
#if defined(_tsan_enabled_)
    const TRefCounter* RefCounter_ = nullptr;

    const TRefCounter* RefCounter() const
    {
        return RefCounter_;
    }
#else
    const TRefCounter* RefCounter() const
    {
        return GetRefCounter(T_);
    }
#endif
};

////////////////////////////////////////////////////////////////////////////////

//! Creates a weak pointer wrapper for a given raw pointer.
//! Compared to |TWeakPtr<T>::ctor|, type inference enables omitting |T|.
template <class T>
TWeakPtr<T> MakeWeak(T* p)
{
    return TWeakPtr<T>(p);
}

//! Creates a weak pointer wrapper for a given intrusive pointer.
//! Compared to |TWeakPtr<T>::ctor|, type inference enables omitting |T|.
template <class T>
TWeakPtr<T> MakeWeak(const TIntrusivePtr<T>& p)
{
    return TWeakPtr<T>(p);
}

//! A helper for acquiring weak pointer for pointee, resetting intrusive pointer and then
//! returning the pointee reference count using the acquired weak pointer.
//! This helper is designed for best effort in checking that the object is not leaked after
//! destructing (what seems to be) the last pointer to it.
//! NB: it is possible to rewrite this helper making it working event with intrinsic refcounted objects,
//! but it requires much nastier integration with the intrusive pointer destruction routines.
template <typename T>
int ResetAndGetResidualRefCount(TIntrusivePtr<T>& pointer)
{
    auto weakPointer = MakeWeak(pointer);
    pointer.Reset();
    pointer = weakPointer.Lock();
    if (pointer) {
        // This _may_ return 0 if we are again the only holder of the pointee.
        return pointer->GetRefCount() - 1;
    } else {
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////

template <class T, class U>
bool operator==(const TWeakPtr<T>& lhs, const TWeakPtr<U>& rhs)
{
    static_assert(
        std::is_convertible_v<U*, T*>,
        "U* must be convertible to T*");
    return lhs.TryGetRefCounter() == rhs.TryGetRefCounter();
}

template <class T, class U>
bool operator!=(const TWeakPtr<T>& lhs, const TWeakPtr<U>& rhs)
{
    static_assert(
        std::is_convertible_v<U*, T*>,
        "U* must be convertible to T*");
    return lhs.TryGetRefCounter() != rhs.TryGetRefCounter();
}

template <class T>
bool operator==(std::nullptr_t, const TWeakPtr<T>& rhs)
{
    return nullptr == rhs.TryGetRefCounter();
}

template <class T>
bool operator!=(std::nullptr_t, const TWeakPtr<T>& rhs)
{
    return nullptr != rhs.TryGetRefCounter();
}

template <class T>
bool operator==(const TWeakPtr<T>& lhs, std::nullptr_t)
{
    return nullptr == lhs.TryGetRefCounter();
}

template <class T>
bool operator!=(const TWeakPtr<T>& lhs, std::nullptr_t)
{
    return nullptr != lhs.TryGetRefCounter();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT


//! A hasher for TWeakPtr.
template <class T>
struct THash<NYT::TWeakPtr<T>>
{
    size_t operator () (const NYT::TWeakPtr<T>& ptr) const
    {
        return THash<const NYT::TRefCountedBase*>()(ptr.T_);
    }
};
