#ifndef WEAK_PTR_INL_H_
#error "Direct inclusion of this file is not allowed, include weak_ptr.h"
// For the sake of sane code completion.
#include "weak_ptr.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
TWeakPtr<T>::TWeakPtr(std::nullptr_t)
{ }

template <class T>
TWeakPtr<T>::TWeakPtr(T* p) noexcept
    : T_(p)
{
#if defined(_tsan_enabled_)
    if (T_) {
        RefCounter_ = GetRefCounter(T_);
    }
#endif
    AcquireRef();
}

template <class T>
TWeakPtr<T>::TWeakPtr(const TIntrusivePtr<T>& ptr) noexcept
    : TWeakPtr(ptr.Get())
{ }

template <class T>
template <class U, class>
TWeakPtr<T>::TWeakPtr(const TIntrusivePtr<U>& ptr) noexcept
    : TWeakPtr(ptr.Get())
{
    static_assert(
        std::derived_from<T, TRefCountedBase>,
        "Cast allowed only for types derived from TRefCountedBase");
}

template <class T>
TWeakPtr<T>::TWeakPtr(const TWeakPtr& other) noexcept
    : T_(other.T_)
#if defined(_tsan_enabled_)
    , RefCounter_(other.RefCounter_)
#endif
{
    AcquireRef();
}

template <class T>
template <class U, class>
TWeakPtr<T>::TWeakPtr(const TWeakPtr<U>& other) noexcept
    : TWeakPtr(other.Lock())
{
    static_assert(
        std::derived_from<T, TRefCountedBase>,
        "Cast allowed only for types derived from TRefCountedBase");
}

template <class T>
TWeakPtr<T>::TWeakPtr(TWeakPtr&& other) noexcept
{
    other.Swap(*this);
}

template <class T>
template <class U, class>
TWeakPtr<T>::TWeakPtr(TWeakPtr<U>&& other) noexcept
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

template <class T>
TWeakPtr<T>::~TWeakPtr()
{
    ReleaseRef();
}

template <class T>
template <class U>
TWeakPtr<T>& TWeakPtr<T>::operator=(const TIntrusivePtr<U>& ptr) noexcept
{
    static_assert(
        std::is_convertible_v<U*, T*>,
        "U* must be convertible to T*");
    TWeakPtr(ptr).Swap(*this);
    return *this;
}

template <class T>
TWeakPtr<T>& TWeakPtr<T>::operator=(const TWeakPtr& other) noexcept
{
    TWeakPtr(other).Swap(*this);
    return *this;
}

template <class T>
template <class U>
TWeakPtr<T>& TWeakPtr<T>::operator=(const TWeakPtr<U>& other) noexcept
{
    static_assert(
        std::is_convertible_v<U*, T*>,
        "U* must be convertible to T*");
    TWeakPtr(other).Swap(*this);
    return *this;
}

template <class T>
TWeakPtr<T>& TWeakPtr<T>::operator=(TWeakPtr&& other) noexcept
{
    other.Swap(*this);
    return *this;
}

template <class T>
template <class U>
TWeakPtr<T>& TWeakPtr<T>::operator=(TWeakPtr<U>&& other) noexcept
{
    static_assert(
        std::is_convertible_v<U*, T*>,
        "U* must be convertible to T*");
    TWeakPtr(std::move(other)).Swap(*this);
    return *this;
}

template <class T>
void TWeakPtr<T>::Reset() // noexcept
{
    TWeakPtr().Swap(*this);
}

template <class T>
void TWeakPtr<T>::Reset(T* p) // noexcept
{
    TWeakPtr(p).Swap(*this);
}

template <class T>
template <class U>
void TWeakPtr<T>::Reset(const TIntrusivePtr<U>& ptr) // noexcept
{
    static_assert(
        std::is_convertible_v<U*, T*>,
        "U* must be convertible to T*");
    TWeakPtr(ptr).Swap(*this);
}

template <class T>
void TWeakPtr<T>::Swap(TWeakPtr& other) noexcept
{
    DoSwap(T_, other.T_);
#if defined(_tsan_enabled_)
    DoSwap(RefCounter_, other.RefCounter_);
#endif
}

template <class T>
TIntrusivePtr<T> TWeakPtr<T>::Lock() const noexcept
{
    return T_ && GetRefCounterImpl()->TryRef()
        ? TIntrusivePtr<T>(T_, false)
        : TIntrusivePtr<T>();
}

template <class T>
bool TWeakPtr<T>::IsExpired() const noexcept
{
    return !T_ || (GetRefCounterImpl()->GetRefCount() == 0);
}

template <class T>
T* TWeakPtr<T>::Get() const noexcept
{
    return T_;
}

template <class T>
const TRefCounter* TWeakPtr<T>::TryGetRefCounterImpl() const
{
    return T_ ? GetRefCounterImpl() : nullptr;
}

template <class T>
void TWeakPtr<T>::AcquireRef()
{
    if (T_) {
        GetRefCounterImpl()->WeakRef();
    }
}

template <class T>
void TWeakPtr<T>::ReleaseRef()
{
    if (T_) {
        // Support incomplete type.
        if (GetRefCounterImpl()->WeakUnref()) {
            DeallocateRefCounted(T_);
        }
    }
}

#if defined(_tsan_enabled_)
template <class T>
const TRefCounter* TWeakPtr<T>::GetRefCounterImpl() const
{
    return RefCounter_;
}
#else
template <class T>
const TRefCounter* TWeakPtr<T>::GetRefCounterImpl() const
{
    return GetRefCounter(T_);
}
#endif

////////////////////////////////////////////////////////////////////////////////

template <class T1, class T2>
bool operator==(const TWeakPtr<T1>& lhs, const TWeakPtr<T2>& rhs)
{
    return lhs.Get() == rhs.Get();
}

template <class T1, class T2>
bool operator==(const TIntrusivePtr<T1>& lhs, const TWeakPtr<T2>& rhs)
{
    return lhs.Get() == rhs.Get();
}

template <class T1, class T2>
bool operator==(const TWeakPtr<T1>& lhs, const TIntrusivePtr<T2>& rhs)
{
    return lhs.Get() == rhs.Get();
}

template <class T1, class T2>
bool operator==(T1* lhs, const TWeakPtr<T2>& rhs)
{
    return lhs == rhs.Get();
}

template <class T1, class T2>
bool operator==(const TWeakPtr<T1>& lhs, T2* rhs)
{
    return lhs.Get() == rhs;
}

template <class T2>
bool operator==(std::nullptr_t, const TWeakPtr<T2>& rhs)
{
    return nullptr == rhs.Get();
}

template <class T1>
bool operator==(const TWeakPtr<T1>& lhs, std::nullptr_t)
{
    return lhs.Get() == nullptr;
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
TWeakPtr<T> MakeWeak(T* p)
{
    return TWeakPtr<T>(p);
}

template <class T>
TWeakPtr<T> MakeWeak(const TIntrusivePtr<T>& p)
{
    return TWeakPtr<T>(p);
}

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

template <class T>
std::size_t TTransparentWeakPtrHasher::operator()(const TWeakPtr<T>& ptr) const
{
    return THash<const NYT::TRefCountedBase*>()(ptr.Get());
}

template <class T>
std::size_t TTransparentWeakPtrHasher::operator()(const TIntrusivePtr<T>& ptr) const
{
    return THash<const NYT::TRefCountedBase*>()(ptr.Get());
}

template <class T>
std::size_t TTransparentWeakPtrHasher::operator()(T* ptr) const
{
    return THash<const NYT::TRefCountedBase*>()(ptr);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

////////////////////////////////////////////////////////////////////////////////

//! A hasher for TWeakPtr.
template <class T>
size_t THash<NYT::TWeakPtr<T>>::operator()(const NYT::TWeakPtr<T>& ptr) const
{
    return THash<const NYT::TRefCountedBase*>()(ptr.Get());
}

////////////////////////////////////////////////////////////////////////////////
