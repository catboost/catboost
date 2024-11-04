#pragma once

#include "ref_counted.h"

#include <util/generic/hash.h>
#include <util/generic/utility.h>

#include <utility>
#include <type_traits>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
class TIntrusivePtr
{
public:
    using TUnderlying = T;

    //! For compatibility with std:: smart pointers.
    using element_type = T;

    constexpr TIntrusivePtr() noexcept
    { }

    constexpr TIntrusivePtr(std::nullptr_t) noexcept
    { }

    //! Constructor from an unqualified reference.
    /*!
     * Note that this constructor could be racy due to unsynchronized operations
     * on the object and on the counter.
     *
     * Note that it notoriously hard to make this constructor explicit
     * given the current amount of code written.
     */
    TIntrusivePtr(T* obj, bool addReference = true) noexcept
        : T_(obj)
    {
        if (T_ && addReference) {
            Ref(T_);
        }
    }

    //! Copy constructor.
    TIntrusivePtr(const TIntrusivePtr& other) noexcept
        : T_(other.Get())
    {
        if (T_) {
            Ref(T_);
        }
    }

    //! Copy constructor with an upcast.
    template <class U, class = typename std::enable_if_t<std::is_convertible_v<U*, T*>>>
    TIntrusivePtr(const TIntrusivePtr<U>& other) noexcept
        : T_(other.Get())
    {
        static_assert(
            std::derived_from<T, TRefCountedBase>,
            "Cast allowed only for types derived from TRefCountedBase");
        if (T_) {
            Ref(T_);
        }
    }

    //! Move constructor.
    TIntrusivePtr(TIntrusivePtr&& other) noexcept
        : T_(other.Get())
    {
        other.T_ = nullptr;
    }

    //! Move constructor with an upcast.
    template <class U, class = typename std::enable_if_t<std::is_convertible_v<U*, T*>>>
    TIntrusivePtr(TIntrusivePtr<U>&& other) noexcept
        : T_(other.Get())
    {
        static_assert(
            std::derived_from<T, TRefCountedBase>,
            "Cast allowed only for types derived from TRefCountedBase");
        other.T_ = nullptr;
    }

    //! Destructor.
    ~TIntrusivePtr()
    {
        if (T_) {
            Unref(T_);
        }
    }

    //! Copy assignment operator.
    TIntrusivePtr& operator=(const TIntrusivePtr& other) noexcept
    {
        TIntrusivePtr(other).Swap(*this);
        return *this;
    }

    //! Copy assignment operator with an upcast.
    template <class U>
    TIntrusivePtr& operator=(const TIntrusivePtr<U>& other) noexcept
    {
        static_assert(
            std::is_convertible_v<U*, T*>,
            "U* must be convertible to T*");
        static_assert(
            std::derived_from<T, TRefCountedBase>,
            "Cast allowed only for types derived from TRefCountedBase");
        TIntrusivePtr(other).Swap(*this);
        return *this;
    }

    //! Move assignment operator.
    TIntrusivePtr& operator=(TIntrusivePtr&& other) noexcept
    {
        TIntrusivePtr(std::move(other)).Swap(*this);
        return *this;
    }

    //! Move assignment operator with an upcast.
    template <class U>
    TIntrusivePtr& operator=(TIntrusivePtr<U>&& other) noexcept
    {
        static_assert(
            std::is_convertible_v<U*, T*>,
            "U* must be convertible to T*");
        static_assert(
            std::derived_from<T, TRefCountedBase>,
            "Cast allowed only for types derived from TRefCountedBase");
        TIntrusivePtr(std::move(other)).Swap(*this);
        return *this;
    }

    //! Drop the pointer.
    void Reset() // noexcept
    {
        TIntrusivePtr().Swap(*this);
    }

    //! Replace the pointer with a specified one.
    void Reset(T* p) // noexcept
    {
        TIntrusivePtr(p).Swap(*this);
    }

    //! Returns the pointer.
    T* Get() const noexcept
    {
        return T_;
    }

    //! Returns the pointer, for compatibility with std:: smart pointers.
    T* get() const noexcept
    {
        return T_;
    }

    //! Returns the pointer and releases the ownership.
    T* Release() noexcept
    {
        auto* p = T_;
        T_ = nullptr;
        return p;
    }

    T& operator*() const noexcept
    {
        YT_ASSERT(T_);
        return *T_;
    }

    T* operator->() const noexcept
    {
        YT_ASSERT(T_);
        return T_;
    }

    explicit operator bool() const noexcept
    {
        return T_ != nullptr;
    }

    //! Swap the pointer with the other one.
    void Swap(TIntrusivePtr& r) noexcept
    {
        DoSwap(T_, r.T_);
    }

private:
    template <class U>
    friend class TIntrusivePtr;

    T* T_ = nullptr;
};

////////////////////////////////////////////////////////////////////////////////

//! Creates a strong pointer wrapper for a given raw pointer.
//! Compared to |TIntrusivePtr<T>::ctor|, type inference enables omitting |T|.
template <class T>
TIntrusivePtr<T> MakeStrong(T* p)
{
    return TIntrusivePtr<T>(p);
}

//! Tries to obtain an intrusive pointer for an object that may had
//! already lost all of its references and, thus, is about to be deleted.
/*!
 * You may call this method at any time provided that you have a valid
 * raw pointer to an object. The call either returns an intrusive pointer
 * for the object (thus ensuring that the object won't be destroyed until
 * you're holding this pointer) or NULL indicating that the last reference
 * had already been lost and the object is on its way to heavens.
 * All these steps happen atomically.
 *
 * Under all circumstances it is caller's responsibility the make sure that
 * the object is not destroyed during the call to #DangerousGetPtr.
 * Typically this is achieved by keeping a (lock-protected) collection of
 * raw pointers, taking a lock in object's destructor, and unregistering
 * its raw pointer from the collection there.
 */

template <class T>
Y_FORCE_INLINE TIntrusivePtr<T> DangerousGetPtr(T* object)
{
    return object->TryRef()
        ? TIntrusivePtr<T>(object, false)
        : TIntrusivePtr<T>();
}

////////////////////////////////////////////////////////////////////////////////

template <class T, class U>
TIntrusivePtr<T> StaticPointerCast(const TIntrusivePtr<U>& ptr)
{
    return {static_cast<T*>(ptr.Get())};
}

template <class T, class U>
TIntrusivePtr<T> StaticPointerCast(TIntrusivePtr<U>&& ptr)
{
    return {static_cast<T*>(ptr.Release()), false};
}

template <class T, class U>
TIntrusivePtr<T> ConstPointerCast(const TIntrusivePtr<U>& ptr)
{
    return {const_cast<T*>(ptr.Get())};
}

template <class T, class U>
TIntrusivePtr<T> ConstPointerCast(TIntrusivePtr<U>&& ptr)
{
    return {const_cast<T*>(ptr.Release()), false};
}

template <class T, class U>
TIntrusivePtr<T> DynamicPointerCast(const TIntrusivePtr<U>& ptr)
{
    return {dynamic_cast<T*>(ptr.Get())};
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
bool operator<(const TIntrusivePtr<T>& lhs, const TIntrusivePtr<T>& rhs)
{
    return lhs.Get() < rhs.Get();
}

template <class T>
bool operator>(const TIntrusivePtr<T>& lhs, const TIntrusivePtr<T>& rhs)
{
    return lhs.Get() > rhs.Get();
}

template <class T, class U>
bool operator==(const TIntrusivePtr<T>& lhs, const TIntrusivePtr<U>& rhs)
{
    static_assert(
        std::is_convertible_v<U*, T*>,
        "U* must be convertible to T*");
    return lhs.Get() == rhs.Get();
}

template <class T, class U>
bool operator!=(const TIntrusivePtr<T>& lhs, const TIntrusivePtr<U>& rhs)
{
    static_assert(
        std::is_convertible_v<U*, T*>,
        "U* must be convertible to T*");
    return lhs.Get() != rhs.Get();
}

template <class T, class U>
bool operator==(const TIntrusivePtr<T>& lhs, U* rhs)
{
    return lhs.Get() == rhs;
}

template <class T, class U>
bool operator!=(const TIntrusivePtr<T>& lhs, U* rhs)
{
    static_assert(
        std::is_convertible_v<U*, T*>,
        "U* must be convertible to T*");
    return lhs.Get() != rhs;
}

template <class T, class U>
bool operator==(T* lhs, const TIntrusivePtr<U>& rhs)
{
    return lhs == rhs.Get();
}

template <class T, class U>
bool operator!=(T* lhs, const TIntrusivePtr<U>& rhs)
{
    static_assert(
        std::is_convertible_v<U*, T*>,
        "U* must be convertible to T*");
    return lhs != rhs.Get();
}

template <class T>
bool operator==(std::nullptr_t, const TIntrusivePtr<T>& rhs)
{
    return nullptr == rhs.Get();
}

template <class T>
bool operator!=(std::nullptr_t, const TIntrusivePtr<T>& rhs)
{
    return nullptr != rhs.Get();
}

template <class T>
bool operator==(const TIntrusivePtr<T>& lhs, std::nullptr_t)
{
    return nullptr == lhs.Get();
}

template <class T>
bool operator!=(const TIntrusivePtr<T>& lhs, std::nullptr_t)
{
    return nullptr != lhs.Get();
}

////////////////////////////////////////////////////////////////////////////////

} //namespace NYT

//! A hasher for TIntrusivePtr.
template <class T>
struct THash<NYT::TIntrusivePtr<T>>
{
    Y_FORCE_INLINE size_t operator () (const NYT::TIntrusivePtr<T>& ptr) const
    {
        return THash<T*>()(ptr.Get());
    }
};
