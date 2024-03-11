#ifndef WRAPPER_TRAITS_INL_H_
#error "Direct inclusion of this file is not allowed, include wrapper_traits.h"
// For the sake of sane code completion.
#include "wrapper_traits.h"
#endif

#include <library/cpp/yt/assert/assert.h>

#include <optional>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
    requires std::is_object_v<T>
template <class U>
    requires std::same_as<std::remove_cvref_t<U>, T>
constexpr decltype(auto) TBasicWrapperTraits<T>::Unwrap(U&& wrapper) noexcept
{
    return std::forward<U>(wrapper);
}

template <class T>
    requires std::is_object_v<T>
constexpr bool TBasicWrapperTraits<T>::HasValue(const T&) noexcept
{
    return true;
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
    requires std::is_object_v<T>
template <class U>
    requires std::same_as<std::remove_cvref_t<U>, T>
constexpr decltype(auto) TWrapperTraits<T>::Unwrap(U&& wrapper)
{
    YT_VERIFY(HasValue(wrapper));
    return TBasicWrapperTraits<T>::Unwrap(std::forward<U>(wrapper));
}

template <class T>
    requires std::is_object_v<T>
template <class U>
    requires std::same_as<std::remove_cvref_t<U>, T>
constexpr decltype(auto) TWrapperTraits<T>::RecursiveUnwrap(U&& wrapper)
{
    using TDecayedU = std::remove_cvref_t<U>;
    using TTraits = TWrapperTraits<TDecayedU>;
    using TInnerTraits = TWrapperTraits<typename TTraits::TUnwrapped>;

    if constexpr (CNonTrivialWrapper<TDecayedU>) {
        return TInnerTraits::RecursiveUnwrap(TTraits::Unwrap(std::forward<U>(wrapper)));
    } else {
        return TTraits::Unwrap(std::forward<U>(wrapper));
    }
}

template <class T>
    requires std::is_object_v<T>
constexpr bool TWrapperTraits<T>::HasValue(const T& wrapper) noexcept
{
    return TBasicWrapperTraits<T>::HasValue(wrapper);
}

template <class T>
    requires std::is_object_v<T>
constexpr bool TWrapperTraits<T>::RecursiveHasValue(const T& wrapper) noexcept
{
    using TTraits = TWrapperTraits<T>;
    using TInnerTraits = TWrapperTraits<typename TTraits::TUnwrapped>;

    if constexpr (CNonTrivialWrapper<T>) {
        return TTraits::HasValue(wrapper) && TInnerTraits::HasValue(TTraits::Unwrap(wrapper));
    } else {
        return TTraits::HasValue(wrapper);
    }
}

template <class T>
    requires std::is_object_v<T>
template <class U>
constexpr T TWrapperTraits<T>::Wrap(U&& unwrapped) noexcept
{
    static_assert(std::same_as<std::remove_cvref_t<U>, TUnwrapped>);

    return T(std::forward<U>(unwrapped));
}

template <class T>
    requires std::is_object_v<T>
template <class U>
constexpr T TWrapperTraits<T>::RecursiveWrap(U&& unwrapped) noexcept
{
    using TTraits = TWrapperTraits<T>;
    using TInnerTraits = TWrapperTraits<typename TTraits::TUnwrapped>;

    if constexpr (CNonTrivialWrapper<TUnwrapped>) {
        return TTraits::Wrap(TInnerTraits::RecursiveWrap(std::forward<U>(unwrapped)));
    } else {
        //! U == TUnwrapped.
        return TTraits::Wrap(std::forward<U>(unwrapped));
    }
}

////////////////////////////////////////////////////////////////////////////////

//! Some standard library specializations

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TBasicWrapperTraits<std::optional<T>>
{
    static constexpr bool IsTrivialWrapper = false;

    using TUnwrapped = T;

    static constexpr bool HasValue(const std::optional<T>& optional) noexcept
    {
        return optional.has_value();
    }

    template <class U>
        requires std::same_as<std::remove_cvref_t<U>, std::optional<T>>
    static constexpr decltype(auto) Unwrap(U&& optional)
    {
        return *std::forward<U>(optional);
    }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
