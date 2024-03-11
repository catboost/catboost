#pragma once

#include <utility>

#include <util/generic/string.h>

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

    #define XX(op) \
        constexpr auto operator op(const TStrongTypedef& rhs) const \
            noexcept(noexcept(Underlying_ op rhs.Underlying_)) \
                requires requires(T lhs, T rhs) {lhs op rhs; };

    XX(<)
    XX(>)
    XX(<=)
    XX(>=)
    XX(==)
    XX(!=)
    XX(<=>)

    #undef XX

    explicit operator bool() const
        noexcept(noexcept(static_cast<bool>(Underlying_)));

    constexpr T& Underlying() &;
    constexpr const T& Underlying() const &;
    constexpr T&& Underlying() &&;

private:
    T Underlying_;

    //! NB: Hidden friend definition to make this name accessible only via ADL.
    friend TString ToString(const TStrongTypedef& value)
        requires requires (T value) { { ToString(value) } -> std::same_as<TString>; }
    {
        return ToString(value.Underlying_);
    }
};

#define YT_DEFINE_STRONG_TYPEDEF(T, TUnderlying) \
    struct T ## Tag \
    { }; \
    using T = ::NYT::TStrongTypedef<TUnderlying, T##Tag>; \

template <class T>
struct TStrongTypedefTraits;

template <class T>
concept CStrongTypedef = TStrongTypedefTraits<T>::IsStrongTypedef;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define STRONG_TYPEDEF_INL_H_
#include "strong_typedef-inl.h"
#undef STRONG_TYPEDEF_INL_H_
