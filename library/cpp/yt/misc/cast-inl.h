#ifndef CAST_INL_H_
#error "Direct inclusion of this file is not allowed, include cast.h"
// For the sake of sane code completion.
#include "cast.h"
#endif

#include "enum.h"

#include <util/string/cast.h>
#include <util/string/printf.h>

#include <type_traits>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class T, class S>
constexpr bool IsInIntegralRange(S value)
    requires std::is_signed_v<T> && std::is_signed_v<S>
{
    return value >= std::numeric_limits<T>::lowest() && value <= std::numeric_limits<T>::max();
}

template <class T, class S>
constexpr bool IsInIntegralRange(S value)
    requires std::is_signed_v<T> && std::is_unsigned_v<S>
{
    return value <= static_cast<typename std::make_unsigned<T>::type>(std::numeric_limits<T>::max());
}

template <class T, class S>
constexpr bool IsInIntegralRange(S value)
    requires std::is_unsigned_v<T> && std::is_signed_v<S>
{
    return value >= 0 && static_cast<typename std::make_unsigned<S>::type>(value) <= std::numeric_limits<T>::max();
}

template <class T, class S>
constexpr bool IsInIntegralRange(S value)
    requires std::is_unsigned_v<T> && std::is_unsigned_v<S>
{
    return value <= std::numeric_limits<T>::max();
}

template <class T, class S>
constexpr bool IsInIntegralRange(S value)
    requires std::is_enum_v<S>
{
    return IsInIntegralRange<T>(static_cast<std::underlying_type_t<S>>(value));
}

template <class T>
TString FormatInvalidCastValue(T value)
{
    return ::ToString(value);
}

inline TString FormatInvalidCastValue(signed char value)
{
    return TString("'") + value + TString("'");
}

inline TString FormatInvalidCastValue(unsigned char value)
{
    return TString("'") + value + TString("'");
}

#ifdef __cpp_char8_t
inline TString FormatInvalidCastValue(char8_t value)
{
    return FormatInvalidCastValue(static_cast<unsigned char>(value));
}
#endif

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////


template <class T, class S>
constexpr bool CanFitSubtype()
{
    return NYT::NDetail::IsInIntegralRange<T>(std::numeric_limits<S>::min()) &&
        NYT::NDetail::IsInIntegralRange<T>(std::numeric_limits<S>::max());
}

template <class T, class S>
constexpr bool IsInIntegralRange(S value)
{
    return NYT::NDetail::IsInIntegralRange<T>(value);
}

template <class T, class S>
constexpr std::optional<T> TryCheckedIntegralCast(S value)
{
    [[unlikely]] if (!NYT::NDetail::IsInIntegralRange<T>(value)) {
        return std::nullopt;
    }
    return static_cast<T>(value);
}

template <class T, class S>
T CheckedIntegralCast(S value)
{
    auto result = TryCheckedIntegralCast<T>(value);
    if (!result) {
        throw TSimpleException(Sprintf("Error casting %s value \"%s\" to %s: value is out of expected range [%s; %s]",
            TypeName<S>().c_str(),
            NYT::NDetail::FormatInvalidCastValue(value).c_str(),
            TypeName<T>().c_str(),
            ::ToString(std::numeric_limits<T>::lowest()).c_str(),
            ::ToString(std::numeric_limits<T>::max()).c_str()));
    }
    return *result;
}

template <class T, class S>
    requires TEnumTraits<T>::IsEnum
constexpr std::optional<T> TryCheckedEnumCast(S value, bool enableUnknown)
{
    auto underlying = TryCheckedIntegralCast<std::underlying_type_t<T>>(value);
    [[unlikely]] if (!underlying) {
        return std::nullopt;
    }
    auto candidate = static_cast<T>(*underlying);
    [[unlikely]] if (!TEnumTraits<T>::IsValidValue(candidate)) {
        if (enableUnknown) {
            if constexpr (constexpr auto optionalUnknownValue = TEnumTraits<T>::TryGetUnknownValue()) {
                if constexpr (TEnumTraits<T>::IsBitEnum) {
                    return static_cast<T>(*underlying & ToUnderlying(TEnumTraits<T>::GetAllSetValue())) | *optionalUnknownValue;
                } else {
                    return *optionalUnknownValue;
                }
            }
        }
        return std::nullopt;
    }
    return candidate;
}

template <class T, class S>
    requires TEnumTraits<T>::IsEnum
T CheckedEnumCast(S value)
{
    auto result = TryCheckedEnumCast<T>(value, /*enableUnknown*/ true);
    [[unlikely]] if (!result) {
        if constexpr (std::is_signed_v<S>) {
            throw TSimpleException(Sprintf("Error casting %s value \"%" PRIi64 "\" to enum %s",
                TypeName<S>().c_str(),
                static_cast<i64>(value),
                TEnumTraits<T>::GetTypeName().data()));
        } else {
            throw TSimpleException(Sprintf("Error casting %s value \"%" PRIu64 "\" to enum %s",
                TypeName<S>().c_str(),
                static_cast<ui64>(value),
                TEnumTraits<T>::GetTypeName().data()));
        }
    }
    return *result;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
