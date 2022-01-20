#pragma once

#include "string.h"

#include <library/cpp/yt/misc/enum.h>

#include <optional>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

TString DecodeEnumValue(TStringBuf value);
TString EncodeEnumValue(TStringBuf value);

template <class T>
std::optional<T> TryParseEnum(TStringBuf value)
{
    static_assert(TEnumTraits<T>::IsEnum);

    auto tryFromString = [] (TStringBuf value) -> std::optional<T> {
        T result;
        if (auto ok = TEnumTraits<T>::FindValueByLiteral(DecodeEnumValue(value), &result)) {
            return result;
        }
        return {};
    };

    if constexpr (TEnumTraits<T>::IsBitEnum) {
        T result = {};
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

[[noreturn]] void ThrowEnumParsingError(TStringBuf name, TStringBuf value);

template <class T>
T ParseEnum(TStringBuf value)
{
    if (auto optionalResult = TryParseEnum<T>(value)) {
        return *optionalResult;
    }
    ThrowEnumParsingError(
        TEnumTraits<T>::GetTypeName(),
        value);
}

void FormatUnknownEnum(TStringBuilderBase* builder, TStringBuf name, i64 value);

template <class T>
void FormatEnum(TStringBuilderBase* builder, T value, bool lowerCase)
{
    static_assert(TEnumTraits<T>::IsEnum);

    auto formatScalarValue = [builder, lowerCase] (T value) {
        auto* literal = TEnumTraits<T>::FindLiteralByValue(value);
        if (!literal) {
            YT_VERIFY(!TEnumTraits<T>::IsBitEnum);
            FormatUnknownEnum(
                builder,
                TEnumTraits<T>::GetTypeName(),
                static_cast<typename TEnumTraits<T>::TUnderlying>(value));
            return;
        }

        if (lowerCase) {
            CamelCaseToUnderscoreCase(builder, *literal);
        } else {
            builder->AppendString(*literal);
        }
    };

    if constexpr (TEnumTraits<T>::IsBitEnum) {
        auto* literal = TEnumTraits<T>::FindLiteralByValue(value);
        if (literal) {
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
TString FormatEnum(T value, typename TEnumTraits<T>::TType* = nullptr)
{
    TStringBuilder b;
    FormatEnum(&b, value, /* lowerCase */ true);
    return b.Flush();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
