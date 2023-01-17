#ifndef CAST_INL_H_
#error "Direct inclusion of this file is not allowed, include cast.h"
// For the sake of sane code completion.
#include "cast.h"
#endif

#include <util/string/cast.h>
#include <util/string/printf.h>

#include <type_traits>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class T, class S>
typename std::enable_if<std::is_signed<T>::value && std::is_signed<S>::value, bool>::type IsInIntegralRange(S value)
{
    return value >= std::numeric_limits<T>::min() && value <= std::numeric_limits<T>::max();
}

template <class T, class S>
static typename std::enable_if<std::is_signed<T>::value && std::is_unsigned<S>::value, bool>::type IsInIntegralRange(S value)
{
    return value <= static_cast<typename std::make_unsigned<T>::type>(std::numeric_limits<T>::max());
}

template <class T, class S>
static typename std::enable_if<std::is_unsigned<T>::value && std::is_signed<S>::value, bool>::type IsInIntegralRange(S value)
{
    return value >= 0 && static_cast<typename std::make_unsigned<S>::type>(value) <= std::numeric_limits<T>::max();
}

template <class T, class S>
typename std::enable_if<std::is_unsigned<T>::value && std::is_unsigned<S>::value, bool>::type IsInIntegralRange(S value)
{
    return value <= std::numeric_limits<T>::max();
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

template <class T, class S>
bool TryIntegralCast(S value, T* result)
{
    if (!NYT::NDetail::IsInIntegralRange<T>(value)) {
        return false;
    }
    *result = static_cast<T>(value);
    return true;
}

template <class T, class S>
T CheckedIntegralCast(S value)
{
    T result;
    if (!TryIntegralCast<T>(value, &result)) {
        throw TSimpleException(Sprintf("Argument value %s is out of expected range",
            NYT::NDetail::FormatInvalidCastValue(value).c_str()));
    }
    return result;
}

template <class T, class S>
bool TryEnumCast(S value, T* result)
{
    auto candidate = static_cast<T>(value);
    if (!TEnumTraits<T>::FindLiteralByValue(candidate)) {
        return false;
    }
    *result = candidate;
    return true;
}

template <class T, class S>
T CheckedEnumCast(S value)
{
    T result;
    if (!TryEnumCast<T>(value, &result)) {
        throw TSimpleException(Sprintf("Invalid value %d of enum type %s",
            static_cast<int>(value),
            TEnumTraits<T>::GetTypeName().data()));
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
