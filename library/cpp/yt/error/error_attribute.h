#pragma once

#include "public.h"

#include <library/cpp/yt/misc/guid.h>
#include <library/cpp/yt/misc/tag_invoke_cpo.h>

#include <string>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
concept CPrimitiveConvertible =
    std::same_as<T, i8> ||
    std::same_as<T, i32> ||
    std::same_as<T, i64> ||
    std::same_as<T, ui8> ||
    std::same_as<T, ui32> ||
    std::same_as<T, ui64> ||
    std::same_as<T, float> ||
    std::same_as<T, double> ||
    std::constructible_from<TStringBuf, const T&> ||
    std::same_as<T, TDuration> ||
    std::same_as<T, TInstant> ||
    std::same_as<T, bool> ||
    std::same_as<T, TGuid>;

////////////////////////////////////////////////////////////////////////////////

namespace NAttributeValueConversionImpl {

struct TTo
    : public TTagInvokeCpoBase<TTo>
{ };

////////////////////////////////////////////////////////////////////////////////

template <class U>
struct TFrom
    : public TTagInvokeCpoBase<TFrom<U>>
{ };

} // namespace NAttributeValueConversionImpl

////////////////////////////////////////////////////////////////////////////////

inline constexpr NAttributeValueConversionImpl::TTo ToErrorAttributeValue = {};
template <class U>
inline constexpr NAttributeValueConversionImpl::TFrom<U> FromErrorAttributeValue = {};

////////////////////////////////////////////////////////////////////////////////

template <class T>
concept CConvertibleToAttributeValue = requires (const T& value) {
    { NYT::ToErrorAttributeValue(value) } -> std::same_as<std::string>;
};

template <class T>
concept CConvertibleFromAttributeValue = requires (TStringBuf value) {
    { NYT::FromErrorAttributeValue<T>(value) } -> std::same_as<T>;
};

////////////////////////////////////////////////////////////////////////////////

struct TErrorAttribute
{
    using TKey = std::string;
    using TValue = std::string;

    template <CConvertibleToAttributeValue T>
    TErrorAttribute(const TKey& key, const T& value)
        : Key(key)
        , Value(NYT::ToErrorAttributeValue(value))
    { }

    TKey Key;
    TValue Value;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ERROR_ATTRIBUTE_INL_H_
#include "error_attribute-inl.h"
#undef ERROR_ATTRIBUTE_INL_H_
