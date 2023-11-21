#pragma once

#include <utility>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T, class TTag>
class TStrongTypedef
{
public:
    using TUnderlying = T;

    constexpr TStrongTypedef()
        requires std::is_default_constructible_v<T>;

    constexpr explicit TStrongTypedef(const T& underlying)
        requires std::is_copy_constructible_v<T>;

    constexpr explicit TStrongTypedef(T&& underlying)
        requires std::is_move_constructible_v<T>;

    TStrongTypedef(const TStrongTypedef&) = default;
    TStrongTypedef(TStrongTypedef&&) = default;

    TStrongTypedef& operator=(const TStrongTypedef&) = default;
    TStrongTypedef& operator=(TStrongTypedef&&) = default;

    constexpr explicit operator const T&() const;
    constexpr explicit operator T&();

    constexpr bool operator==(const TStrongTypedef& rhs) const
        noexcept(std::same_as<T, void> || noexcept(Underlying_ == rhs.Underlying_));

    constexpr auto operator<=>(const TStrongTypedef& rhs) const
        noexcept(std::same_as<T, void> || noexcept(Underlying_ <=> rhs.Underlying_));

    constexpr T& Underlying() &;
    constexpr const T& Underlying() const &;
    constexpr T&& Underlying() &&;

private:
    T Underlying_;
};

#define YT_DEFINE_STRONG_TYPEDEF(T, TUnderlying) \
    struct T ## Tag \
    { }; \
    using T = ::NYT::TStrongTypedef<TUnderlying, T##Tag>; \

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define STRONG_TYPEDEF_INL_H_
#include "strong_typedef-inl.h"
#undef STRONG_TYPEDEF_INL_H_
