#ifndef STRONG_TYPEDEF_INL_H_
#error "Direct inclusion of this file is not allowed, include strong_typedef.h"
// For the sake of sane code completion.
#include "strong_typedef.h"
#endif

#include "wrapper_traits.h"

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

#define XX(op, defaultValue) \
    template <class T, class TTag> \
    constexpr auto TStrongTypedef<T, TTag>::operator op(const TStrongTypedef& rhs) const \
        noexcept(noexcept(Underlying_ op rhs.Underlying_)) \
            requires requires (T lhs, T rhs) {lhs op rhs; } \
    { \
        if constexpr (std::same_as<T, void>) { \
            return defaultValue; \
        } \
        return Underlying_ op rhs.Underlying_; \
    }

XX(<, false)
XX(>, false)
XX(<=, true)
XX(>=, true)
XX(==, true)
XX(!=, false)
XX(<=>, std::strong_ordering::equal)

#undef XX

template <class T, class TTag>
TStrongTypedef<T, TTag>::operator bool() const
    noexcept(noexcept(static_cast<bool>(Underlying_)))
{
    return static_cast<bool>(Underlying_);
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
    requires CStrongTypedef<T>
bool TryFromStringImpl(const TChar* data, size_t size, T& value)
{
    return TryFromString(data, size, value.Underlying());
}

////////////////////////////////////////////////////////////////////////////////

class TStringBuilderBase;

template <class T, class TTag>
void FormatValue(TStringBuilderBase* builder, const TStrongTypedef<T, TTag>& value, TStringBuf format)
    noexcept(noexcept(FormatValue(builder, value.Underlying(), format)))
{
    FormatValue(builder, value.Underlying(), format);
}

////////////////////////////////////////////////////////////////////////////////

template <class T, class TTag>
struct TBasicWrapperTraits<TStrongTypedef<T, TTag>>
{
    static constexpr bool IsTrivialWrapper = false;

    using TUnwrapped = T;

    static constexpr bool HasValue(const TStrongTypedef<T, TTag>&) noexcept
    {
        return true;
    }

    template <class U>
        requires std::same_as<std::remove_cvref_t<U>, TStrongTypedef<T, TTag>>
    static constexpr decltype(auto) Unwrap(U&& wrapper) noexcept
    {
        return std::forward<U>(wrapper).Underlying();
    }
};

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

template <class T, class TTag>
    requires std::numeric_limits<T>::is_specialized
class numeric_limits<NYT::TStrongTypedef<T, TTag>>
{
public:
    #define XX(name) \
        static constexpr decltype(numeric_limits<T>::name) name = numeric_limits<T>::name;

    XX(is_specialized)
    XX(is_signed)
    XX(digits)
    XX(digits10)
    XX(max_digits10)
    XX(is_integer)
    XX(is_exact)
    XX(radix)
    XX(min_exponent)
    XX(min_exponent10)
    XX(max_exponent)
    XX(max_exponent10)
    XX(has_infinity)
    XX(has_quiet_NaN)
    XX(has_signaling_NaN)
    XX(has_denorm)
    XX(has_denorm_loss)
    XX(is_iec559)
    XX(is_bounded)
    XX(is_modulo)
    XX(traps)
    XX(tinyness_before)
    XX(round_style)

    #undef XX

    #define XX(name) \
        static constexpr NYT::TStrongTypedef<T, TTag> name() noexcept \
        { \
            return NYT::TStrongTypedef<T, TTag>(numeric_limits<T>::name()); \
        }

    XX(min)
    XX(max)
    XX(lowest)
    XX(epsilon)
    XX(round_error)
    XX(infinity)
    XX(quiet_NaN)
    XX(signaling_NaN)
    XX(denorm_min)

    #undef XX
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
