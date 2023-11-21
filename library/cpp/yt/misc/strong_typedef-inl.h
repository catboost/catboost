#ifndef STRONG_TYPEDEF_INL_H_
#error "Direct inclusion of this file is not allowed, include strong_typedef.h"
// For the sake of sane code completion.
#include "strong_typedef.h"
#endif

#include <util/generic/strbuf.h>

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

template <class T, class TTag>
constexpr bool TStrongTypedef<T, TTag>::operator==(const TStrongTypedef& rhs) const
    noexcept(std::same_as<T, void> || noexcept(Underlying_ == rhs.Underlying_))
{
    //! NB: We add a constexpr branch to keep constexprness of the function
    //! without making extra specializations explicitly.
    if constexpr (std::same_as<T, void>) {
        return true;
    }

    return Underlying_ == rhs.Underlying_;
}

template <class T, class TTag>
constexpr auto TStrongTypedef<T, TTag>::operator<=>(const TStrongTypedef& rhs) const
    noexcept(std::same_as<T, void> || noexcept(Underlying_ <=> rhs.Underlying_))
{
    //! NB: We add a constexpr branch to keep constexprness of the function
    //! without making extra specializations explicitly.
    if constexpr (std::same_as<T, void>) {
        return std::strong_ordering::equal;
    }

    return Underlying_ <=> rhs.Underlying_;
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TStrongTypedefTraits
{
    constexpr static bool IsStrongTypedef = false;
};

template <class T, class TTag>
struct TStrongTypedefTraits<TStrongTypedef<T, TTag>>
{
    constexpr static bool IsStrongTypedef = true;
    using TUnderlying = T;
};

////////////////////////////////////////////////////////////////////////////////

template <class T, class TChar>
    requires TStrongTypedefTraits<T>::IsStrongTypedef
bool TryFromStringImpl(const TChar* data, size_t size, T& value)
{
    return TryFromString(data, size, value.Underlying());
}

////////////////////////////////////////////////////////////////////////////////

class TStringBuilderBase;

template <class T, class TTag>
void FormatValue(TStringBuilderBase* builder, const TStrongTypedef<T, TTag>& value, TStringBuf format)
{
    FormatValue(builder, value.Underlying(), format);
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

////////////////////////////////////////////////////////////////////////////////

template <class T, class TTag>
struct THash<NYT::TStrongTypedef<T, TTag>>
{
    size_t operator()(const NYT::TStrongTypedef<T, TTag>& value) const
    {
        static constexpr bool IsHashable = requires (T value) {
            { THash<T>()(value) } -> std::same_as<size_t>;
        };

        if constexpr (IsHashable) {
            return THash<T>()(value.Underlying());
        } else {
            return std::hash<T>()(value.Underlying());
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

template <class T, class TTag>
IOutputStream& operator<<(IOutputStream& out, const NYT::TStrongTypedef<T, TTag>& value)
{
    return out << value.Underlying();
}

////////////////////////////////////////////////////////////////////////////////
