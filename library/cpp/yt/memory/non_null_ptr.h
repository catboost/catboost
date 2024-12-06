#pragma once

#include <library/cpp/yt/misc/concepts.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// We have one really strange rule in our codestyle - mutable arguments are passed by pointer.
// But if you are not a fan of making your life indefinite,
// you can use this helper, that will validate that pointer you pass is not null.
template <class T>
class TNonNullPtr;

template <class T>
class TNonNullPtrBase
{
public:
    TNonNullPtrBase(T* ptr) noexcept;
    TNonNullPtrBase(const TNonNullPtrBase& other) = default;

    TNonNullPtrBase(std::nullptr_t) = delete;
    TNonNullPtrBase operator=(const TNonNullPtrBase&) = delete;

    T* operator->() const noexcept;
    T& operator*() const noexcept;

protected:
    T* Ptr_;

    TNonNullPtrBase() noexcept;
};

template <class T>
TNonNullPtr<T> GetPtr(T& ref) noexcept;

template <class T>
class TNonNullPtr
    : public TNonNullPtrBase<T>
{
    using TConstPtr = TNonNullPtr<const T>;
    friend TConstPtr;

    using TNonNullPtrBase<T>::TNonNullPtrBase;

    friend TNonNullPtr<T> GetPtr<T>(T& ref) noexcept;
};

// NB(pogorelov): Method definitions placed in .h file (instead of -inl.h) because of clang16 bug.
// TODO(pogorelov): Move method definitions to helpers-inl.h when new clang will be used.
template <CConst T>
class TNonNullPtr<T>
    : public TNonNullPtrBase<T>
{
    using TMutablePtr = TNonNullPtr<std::remove_const_t<T>>;

    using TNonNullPtrBase<T>::TNonNullPtrBase;

    friend TNonNullPtr<T> GetPtr<T>(T& ref) noexcept;

public:
    TNonNullPtr(TMutablePtr mutPtr) noexcept
        : TNonNullPtrBase<T>()
    {
        TNonNullPtrBase<T>::Ptr_ = mutPtr.Ptr_;
    }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define NON_NULL_PTR_H_
#include "non_null_ptr-inl.h"
#undef NON_NULL_PTR_H_
