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

    //! Default constructor.
    TWeakPtr() = default;

    //! Null constructor.
    TWeakPtr(std::nullptr_t);

    //! Constructor from an unqualified reference.
    /*!
     * Note that this constructor could be racy due to unsynchronized operations
     * on the object and on the counter.
     */
    explicit TWeakPtr(T* p) noexcept;

    //! Constructor from a strong reference.
    TWeakPtr(const TIntrusivePtr<T>& ptr) noexcept;

    //! Constructor from a strong reference with an upcast.
    template <class U, class = typename std::enable_if_t<std::is_convertible_v<U*, T*>>>
    TWeakPtr(const TIntrusivePtr<U>& ptr) noexcept;

    //! Copy constructor.
    TWeakPtr(const TWeakPtr& other) noexcept;

    //! Copy constructor with an upcast.
    template <class U, class = typename std::enable_if_t<std::is_convertible_v<U*, T*>>>
    TWeakPtr(const TWeakPtr<U>& other) noexcept;

    //! Move constructor.
    TWeakPtr(TWeakPtr&& other) noexcept;

    //! Move constructor with an upcast.
    template <class U, class = typename std::enable_if_t<std::is_convertible_v<U*, T*>>>
    TWeakPtr(TWeakPtr<U>&& other) noexcept;

    //! Destructor.
    ~TWeakPtr();

    //! Assignment operator from a strong reference.
    template <class U>
    TWeakPtr& operator=(const TIntrusivePtr<U>& ptr) noexcept;

    //! Copy assignment operator.
    TWeakPtr& operator=(const TWeakPtr& other) noexcept;

    //! Copy assignment operator with an upcast.
    template <class U>
    TWeakPtr& operator=(const TWeakPtr<U>& other) noexcept;

    //! Move assignment operator.
    TWeakPtr& operator=(TWeakPtr&& other) noexcept;

    //! Move assignment operator with an upcast.
    template <class U>
    TWeakPtr& operator=(TWeakPtr<U>&& other) noexcept;

    //! Drop the pointer.
    void Reset(); // noexcept

    //! Replace the pointer with a specified one.
    void Reset(T* p); // noexcept

    //! Replace the pointer with a specified one.
    template <class U>
    void Reset(const TIntrusivePtr<U>& ptr); // noexcept

    //! Swap the pointer with the other one.
    void Swap(TWeakPtr& other) noexcept;

    //! Acquire a strong reference to the pointee and return a strong pointer.
    TIntrusivePtr<T> Lock() const noexcept;

    //! Returns true if the pointee is null or already expired.
    bool IsExpired() const noexcept;

    //! Returns the underlying raw pointer. Note that it may point to an already
    //! expired object.
    T* Get() const noexcept;

private:
    void AcquireRef();
    void ReleaseRef();

    template <class U>
    friend class TWeakPtr;
    template <class U>
    friend struct ::THash;

    T* T_ = nullptr;
#if defined(_tsan_enabled_)
    const TRefCounter* RefCounter_ = nullptr;
#endif

    const TRefCounter* GetRefCounterImpl() const;
    const TRefCounter* TryGetRefCounterImpl() const;
};

////////////////////////////////////////////////////////////////////////////////

template <class T1, class T2>
bool operator==(const TWeakPtr<T1>& lhs, const TWeakPtr<T2>& rhs);

template <class T2>
bool operator==(std::nullptr_t, const TWeakPtr<T2>& rhs);

template <class T1>
bool operator==(const TWeakPtr<T1>& lhs, std::nullptr_t);

////////////////////////////////////////////////////////////////////////////////

//! Creates a weak pointer wrapper for a given raw pointer.
//! Compared to |TWeakPtr<T>::ctor|, type inference enables omitting |T|.
template <class T>
TWeakPtr<T> MakeWeak(T* p);

//! Creates a weak pointer wrapper for a given intrusive pointer.
//! Compared to |TWeakPtr<T>::ctor|, type inference enables omitting |T|.
template <class T>
TWeakPtr<T> MakeWeak(const TIntrusivePtr<T>& p);

//! A helper for acquiring weak pointer for pointee, resetting intrusive pointer and then
//! returning the pointee reference count using the acquired weak pointer.
//! This helper is designed for best effort in checking that the object is not leaked after
//! destructing (what seems to be) the last pointer to it.
//! NB: it is possible to rewrite this helper making it working event with intrinsic refcounted objects,
//! but it requires much nastier integration with the intrusive pointer destruction routines.
template <typename T>
int ResetAndGetResidualRefCount(TIntrusivePtr<T>& pointer);

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

#define WEAK_PTR_INL_H_
#include "weak_ptr-inl.h"
#undef WEAK_PTR_INL_H_
