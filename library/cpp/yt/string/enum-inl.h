#ifndef ENUM_INL_H_
#error "Direct inclusion of this file is not allowed, include enum.h"
// For the sake of sane code completion.
#include "enum.h"
#endif

#include "format.h"

#include <library/cpp/yt/exception/exception.h>

#include <util/string/printf.h>
#include <util/string/strip.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

[[noreturn]]
void ThrowMalformedEnumValueException(
    TStringBuf typeName,
    TStringBuf value);

void FormatUnknownEnumValue(
    TStringBuilderBase* builder,
    TStringBuf name,
    i64 value);

} // namespace NDetail

template <class T>
std::optional<T> TryParseEnum(TStringBuf value)
{
    auto tryFromString = [] (TStringBuf value) -> std::optional<T> {
        if (auto decodedValue = TryDecodeEnumValue(value)) {
            auto enumValue = TEnumTraits<T>::FindValueByLiteral(*decodedValue);
            return enumValue ? enumValue : TEnumTraits<T>::FindValueByLiteral(value);
        }

        auto reportError = [value] () {
            throw TSimpleException(Format("Enum value %Qv is neither in a proper underscore case nor in a format \"%v(123)\"",
                value,
                TEnumTraits<T>::GetTypeName()));
        };

        TStringBuf typeName;
        auto isTypeNameCorrect = value.NextTok('(', typeName) && typeName == TEnumTraits<T>::GetTypeName();
        if (!isTypeNameCorrect) {
            reportError();
        }

        TStringBuf enumValue;
        std::underlying_type_t<T> underlyingValue = 0;
        auto isEnumValueCorrect = value.NextTok(')', enumValue) && TryFromString(enumValue, underlyingValue);
        if (!isEnumValueCorrect) {
            reportError();
        }

        auto isParsingComplete = value.empty();
        if (!isParsingComplete) {
            reportError();
        }

        return static_cast<T>(underlyingValue);
    };

    if constexpr (TEnumTraits<T>::IsBitEnum) {
        T result{};
        TStringBuf token;
        while (value.NextTok('|', token)) {
            if (auto scalar = tryFromString(StripString(token))) {
                result |= *scalar;
            } else {
                return {};
            }
        }
        return result;
    } else {
        return tryFromString(value);
    }
}

template <class T>
T ParseEnum(TStringBuf value)
{
    if (auto optionalResult = TryParseEnum<T>(value)) {
        return *optionalResult;
    }
    NYT::NDetail::ThrowMalformedEnumValueException(TEnumTraits<T>::GetTypeName(), value);
}

template <class T>
void FormatEnum(TStringBuilderBase* builder, T value, bool lowerCase)
{
    auto formatScalarValue = [builder, lowerCase] (T value) {
        auto optionalLiteral = TEnumTraits<T>::FindLiteralByValue(value);
        if (!optionalLiteral) {
            NYT::NDetail::FormatUnknownEnumValue(
                builder,
                TEnumTraits<T>::GetTypeName(),
                ToUnderlying(value));
            return;
        }

        if (lowerCase) {
            CamelCaseToUnderscoreCase(builder, *optionalLiteral);
        } else {
            builder->AppendString(*optionalLiteral);
        }
    };

    if constexpr (TEnumTraits<T>::IsBitEnum) {
        if (TEnumTraits<T>::FindLiteralByValue(value)) {
            formatScalarValue(value);
            return;
        }
        auto first = true;
        for (auto scalarValue : TEnumTraits<T>::GetDomainValues()) {
            if (Any(value & scalarValue)) {
                if (!first) {
                    builder->AppendString(TStringBuf(" | "));
                }
                first = false;
                formatScalarValue(scalarValue);
            }
        }
    } else {
        formatScalarValue(value);
    }
}

template <class T>
TString FormatEnum(T value)
{
    TStringBuilder builder;
    FormatEnum(&builder, value, /*lowerCase*/ true);
    return builder.Flush();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
