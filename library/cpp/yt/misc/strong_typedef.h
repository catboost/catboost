#pragma once

#include "strong_typedef-fwd.h"

#include <utility>

#include <util/generic/string.h>

#include <util/stream/fwd.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T, class TTag, TStrongTypedefOptions Options>
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

    #define XX(returnType, op) \
        constexpr returnType operator op(const TStrongTypedef& rhs) const \
            noexcept(noexcept(Underlying_ op rhs.Underlying_)) \
            requires requires(T lhs, T rhs) { lhs op rhs; } && (Options.IsComparable);

    XX(bool, <)
    XX(bool, >)
    XX(bool, <=)
    XX(bool, >=)
    XX(bool, ==)
    XX(bool, !=)
    XX(auto, <=>)

    #undef XX

    explicit operator bool() const
        noexcept(noexcept(static_cast<bool>(Underlying_)));

    constexpr T& Underlying() &;
    constexpr const T& Underlying() const &;
    constexpr T&& Underlying() &&;

    void Save(IOutputStream* out) const;
    void Load(IInputStream* in);

private:
    T Underlying_;
};

#define YT_DEFINE_STRONG_TYPEDEF(T, TUnderlying, ...) \
    struct T ## Tag \
    { }; \
    using T = ::NYT::TStrongTypedef<TUnderlying, T##Tag __VA_OPT__(, ) __VA_ARGS__>; \
    static_assert(true)

template <class T>
struct TStrongTypedefTraits;

template <class T>
concept CStrongTypedef = TStrongTypedefTraits<T>::IsStrongTypedef;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define STRONG_TYPEDEF_INL_H_
#include "strong_typedef-inl.h"
#undef STRONG_TYPEDEF_INL_H_
