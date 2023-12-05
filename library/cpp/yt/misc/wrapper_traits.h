#pragma once

#include <utility>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Default implementation of wrapper traits you can specialize for your type
template <class T>
    requires std::is_object_v<T>
struct TBasicWrapperTraits
{
    static constexpr bool IsTrivialWrapper = true;

    using TUnwrapped = T;

    //! Default implementations just forward the argument
    template <class U>
        requires std::same_as<std::remove_cvref_t<U>, T>
    static constexpr decltype(auto) Unwrap(U&& wrapper) noexcept;

    static constexpr bool HasValue(const T& wrapper) noexcept;
};

////////////////////////////////////////////////////////////////////////////////

//! Represents common denominator of every single value wrapper
template <class T>
    requires std::is_object_v<T>
struct TWrapperTraits
{
public:
    static constexpr bool IsTrivialWrapper = TBasicWrapperTraits<T>::IsTrivialWrapper;

    using TWrapped = T;
    using TUnwrapped = typename TBasicWrapperTraits<T>::TUnwrapped;

    static constexpr bool HasValue(const T& wrapper) noexcept;

    static constexpr bool RecursiveHasValue(const T& wrapper) noexcept;

    template <class U>
        requires std::same_as<std::remove_cvref_t<U>, T>
    static constexpr decltype(auto) Unwrap(U&& wrapper);

    template <class U>
        requires std::same_as<std::remove_cvref_t<U>, T>
    static constexpr decltype(auto) RecursiveUnwrap(U&& wrapper);

    using TRecursiveUnwrapped = std::remove_cvref_t<decltype(TWrapperTraits<T>::RecursiveUnwrap(std::declval<T>()))>;
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
concept CNonTrivialWrapper =
    (!TBasicWrapperTraits<T>::IsTrivialWrapper) &&
    requires (T& wrapper, const T& const_wrapper) {
        typename TBasicWrapperTraits<T>::TUnwrapped;
        { TBasicWrapperTraits<T>::Unwrap(wrapper) } -> std::same_as<typename TBasicWrapperTraits<T>::TUnwrapped&>;
        { TBasicWrapperTraits<T>::Unwrap(std::move(wrapper)) } -> std::same_as<typename TBasicWrapperTraits<T>::TUnwrapped&&>;
        { TBasicWrapperTraits<T>::Unwrap(const_wrapper) } -> std::same_as<const typename TBasicWrapperTraits<T>::TUnwrapped&>;
        { TBasicWrapperTraits<T>::HasValue(const_wrapper) } -> std::same_as<bool>;
    };

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define WRAPPER_TRAITS_INL_H_
#include "wrapper_traits-inl.h"
#undef WRAPPER_TRAITS_INL_H_
