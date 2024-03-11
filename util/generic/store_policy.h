#pragma once

#include <utility>
#include "ptr.h"

template <class TBase, class TCounter>
struct TWithRefCount: public TBase, public TRefCounted<TWithRefCount<TBase, TCounter>, TCounter> {
    using TBase::TBase;
};

template <class T>
struct TPtrPolicy {
    inline TPtrPolicy(T* t)
        : T_(t)
    {
    }

    inline T* Ptr() noexcept {
        return T_;
    }

    inline const T* Ptr() const noexcept {
        return T_;
    }

    T* T_;
};

template <class T>
struct TEmbedPolicy {
    template <typename... Args, typename = typename std::enable_if<std::is_constructible<T, Args...>::value>::type>
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

    template <typename... Args, typename = typename std::enable_if<std::is_constructible<T, Args...>::value>::type>
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

/**
 * Storage class that can be handy for implementing proxies / adaptors that can
 * accept both lvalues and rvalues. In the latter case it's often required to
 * extend the lifetime of the passed rvalue, and the only option is to store it
 * in your proxy / adaptor.
 *
 * Example usage:
 * \code
 * template<class T>
 * struct TProxy {
 *    TAutoEmbedOrPtrPolicy<T> Value_;
 *    // Your proxy code...
 * };
 *
 * template<class T>
 * TProxy<T> MakeProxy(T&& value) {
 *     // Rvalues are automagically moved-from, and stored inside the proxy.
 *     return {std::forward<T>(value)};
 * }
 * \endcode
 *
 * Look at `Reversed` in `adaptor.h` for real example.
 */
template <class T, bool IsReference = std::is_reference<T>::value>
struct TAutoEmbedOrPtrPolicy: TPtrPolicy<std::remove_reference_t<T>> {
    using TBase = TPtrPolicy<std::remove_reference_t<T>>;

    TAutoEmbedOrPtrPolicy(T& reference)
        : TBase(&reference)
    {
    }
};

template <class T>
struct TAutoEmbedOrPtrPolicy<T, false>: TEmbedPolicy<T> {
    using TBase = TEmbedPolicy<T>;

    TAutoEmbedOrPtrPolicy(T&& object)
        : TBase(std::move(object))
    {
    }
};

template <class T>
using TAtomicRefPolicy = TRefPolicy<T, TAtomicCounter>;

template <class T>
using TSimpleRefPolicy = TRefPolicy<T, TSimpleCounter>;
