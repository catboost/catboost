#ifndef STRONG_TYPEDEF_INL_H_
#error "Direct inclusion of this file is not allowed, include strong_typedef.h"
// For the sake of sane code completion.
#include "strong_typedef.h"
#endif

#include <functional>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T, class TTag>
constexpr TStrongTypedef<T, TTag>::TStrongTypedef()
    requires std::is_default_constructible_v<T>
    : Underlying_{}
{ }

template <class T, class TTag>
constexpr TStrongTypedef<T, TTag>::TStrongTypedef(const T& underlying)
    requires std::is_copy_constructible_v<T>
    : Underlying_(underlying)
{ }

template <class T, class TTag>
constexpr TStrongTypedef<T, TTag>::TStrongTypedef(T&& underlying)
    requires std::is_move_constructible_v<T>
    : Underlying_(std::move(underlying))
{ }

template <class T, class TTag>
constexpr TStrongTypedef<T, TTag>& TStrongTypedef<T, TTag>::operator=(const T& rhs)
    requires std::is_copy_assignable_v<T>
{
    Underlying_ = rhs;
    return *this;
}

template <class T, class TTag>
constexpr TStrongTypedef<T, TTag>& TStrongTypedef<T, TTag>::operator=(T&& rhs)
    requires std::is_move_assignable_v<T>
{
    Underlying_ = std::move(rhs);
    return *this;
}

template <class T, class TTag>
constexpr TStrongTypedef<T, TTag>::operator const T&() const
{
    return Underlying_;
}

template <class T, class TTag>
constexpr TStrongTypedef<T, TTag>::operator T&()
{
    return Underlying_;
}

template <class T, class TTag>
constexpr T& TStrongTypedef<T, TTag>::Underlying() &
{
    return Underlying_;
}

template <class T, class TTag>
constexpr const T& TStrongTypedef<T, TTag>::Underlying() const&
{
    return Underlying_;
}

template <class T, class TTag>
constexpr T&& TStrongTypedef<T, TTag>::Underlying() &&
{
    return std::move(Underlying_);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

namespace std {

////////////////////////////////////////////////////////////////////////////////

template <class T, class TTag>
struct hash<NYT::TStrongTypedef<T, TTag>>
{
    size_t operator()(const NYT::TStrongTypedef<T, TTag>& value) const
    {
        return std::hash<T>()(value.Underlying());
    }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace std
