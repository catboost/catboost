#ifndef ENUM_INL_H_
#error "Direct inclusion of this file is not allowed, include enum.h"
// For the sake of sane code completion.
#include "enum.h"
#endif

#include "format.h"
#include "string.h"
#include "string_builder.h"

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
    auto* builder,
    TStringBuf name,
    auto value)
{
    builder->AppendFormat("%v::unknown-%v", name, ToUnderlying(value));
}

} // namespace NDetail

template <class T>
std::optional<T> TryParseEnum(TStringBuf str, bool enableUnknown)
{
    auto tryParseToken = [&] (TStringBuf token) -> std::optional<T> {
        if (auto optionalValue = TEnumTraits<T>::FindValueByLiteral(token)) {
            return *optionalValue;

        }

        if (auto optionalDecodedValue = TryDecodeEnumValue(token)) {
            if (auto optionalValue = TEnumTraits<T>::FindValueByLiteral(*optionalDecodedValue)) {
                return *optionalValue;
            }
        }

        if (enableUnknown) {
            if constexpr (constexpr auto optionalUnknownValue = TEnumTraits<T>::TryGetUnknownValue()) {
                return *optionalUnknownValue;
            }
        }

        return std::nullopt;
    };

    if constexpr (TEnumTraits<T>::IsBitEnum) {
        T result{};
        TStringBuf token;
        while (str.NextTok('|', token)) {
            if (auto optionalValue = tryParseToken(StripString(token))) {
                result |= *optionalValue;
            } else {
                return {};
            }
        }
        return result;
    } else {
        return tryParseToken(str);
    }
}

template <class T>
T ParseEnum(TStringBuf str)
{
    if (auto optionalResult = TryParseEnum<T>(str, /*enableUnkown*/ true)) {
        return *optionalResult;
    }
    NYT::NDetail::ThrowMalformedEnumValueException(TEnumTraits<T>::GetTypeName(), str);
}

template <class T>
void FormatEnum(TStringBuilderBase* builder, T value, bool lowerCase)
{
    auto formatLiteral = [&] (auto* builder, TStringBuf literal) {
        if (lowerCase) {
            CamelCaseToUnderscoreCase(builder, literal);
        } else {
            builder->AppendString(literal);
        }
    };

    if constexpr (TEnumTraits<T>::IsBitEnum) {
        if (None(value)) {
            // Avoid empty string if possible.
            if (auto optionalLiteral = TEnumTraits<T>::FindLiteralByValue(value)) {
                formatLiteral(builder, *optionalLiteral);
            }
            return;
        }

        TDelimitedStringBuilderWrapper delimitedBuilder(builder, " | ");

        T printedValue{};
        for (auto currentValue : TEnumTraits<T>::GetDomainValues()) {
            // Check if currentValue is viable and non-redunant.
            if ((value & currentValue) == currentValue && (printedValue | currentValue) != printedValue) {
                formatLiteral(&delimitedBuilder, *TEnumTraits<T>::FindLiteralByValue(currentValue));
                printedValue |= currentValue;
            }
        }

        // Handle the remainder.
        if (printedValue != value) {
            NYT::NDetail::FormatUnknownEnumValue(&delimitedBuilder, TEnumTraits<T>::GetTypeName(), value & ~printedValue);
        }
    } else {
        if (auto optionalLiteral = TEnumTraits<T>::FindLiteralByValue(value)) {
            formatLiteral(builder, *optionalLiteral);
            return;
        }

        NYT::NDetail::FormatUnknownEnumValue(builder, TEnumTraits<T>::GetTypeName(), value);
    }
}

template <class T>
std::string FormatEnum(T value)
{
    TStringBuilder builder;
    FormatEnum(&builder, value, /*lowerCase*/ true);
    return builder.Flush();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
