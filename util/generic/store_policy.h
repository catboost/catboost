#pragma once

#include <utility>
#include "ptr.h"

template <class TBase, class TCounter>
struct TWithRefCount: public TBase, public TRefCounted<TWithRefCount<TBase, TCounter>, TCounter> {
    template <typename... Args>
    inline TWithRefCount(Args&&... args)
        : TBase(std::forward<Args>(args)...)
    {
    }
};

template <class T>
struct TPtrPolicy {
    inline TPtrPolicy(const T* t)
        : T_(t)
    {
    }

    inline const T* Ptr() const noexcept {
        return T_;
    }

    const T* T_;
};

template <class T>
struct TEmbedPolicy {
    template <typename... Args>
    inline TEmbedPolicy(Args&&... args)
        : T_(std::forward<Args>(args)...)
    {
    }

    inline T* Ptr() noexcept {
        return &T_;
    }

    inline const T* Ptr() const noexcept {
        return &T_;
    }

    T T_;
};

template <class T, class TCounter>
struct TRefPolicy {
    using THelper = TWithRefCount<T, TCounter>;

    template <typename... Args>
    inline TRefPolicy(Args&&... args)
        : T_(new THelper(std::forward<Args>(args)...))
    {
    }

    inline T* Ptr() noexcept {
        return T_.Get();
    }

    inline const T* Ptr() const noexcept {
        return T_.Get();
    }

    TIntrusivePtr<THelper> T_;
};

template <class T>
using TAtomicRefPolicy = TRefPolicy<T, TAtomicCounter>;

template <class T>
using TSimpleRefPolicy = TRefPolicy<T, TSimpleCounter>;
