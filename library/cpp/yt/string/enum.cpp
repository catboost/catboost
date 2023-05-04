#include "enum.h"

#include "format.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <bool failOnError>
std::optional<TString> DecodeEnumValueImpl(TStringBuf value)
{
    auto camelValue = UnderscoreCaseToCamelCase(value);
    auto underscoreValue = CamelCaseToUnderscoreCase(camelValue);
    if (value != underscoreValue) {
        if constexpr (failOnError) {
            throw TSimpleException(Format("Enum value %Qv is not in a proper underscore case; did you mean %Qv?",
                value,
                underscoreValue));
        } else {
            return std::nullopt;
        }
    }
    return camelValue;
}

std::optional<TString> TryDecodeEnumValue(TStringBuf value)
{
    return DecodeEnumValueImpl<false>(value);
}

TString DecodeEnumValue(TStringBuf value)
{
    auto decodedValue = DecodeEnumValueImpl<true>(value);
    YT_VERIFY(decodedValue);
    return *decodedValue;
}

TString EncodeEnumValue(TStringBuf value)
{
    return CamelCaseToUnderscoreCase(value);
}

namespace NDetail {

void ThrowMalformedEnumValueException(TStringBuf typeName, TStringBuf value)
{
    throw TSimpleException(Format("Error parsing %v value %Qv",
        typeName,
        value));
}

void FormatUnknownEnumValue(TStringBuilderBase* builder, TStringBuf name, i64 value)
{
    builder->AppendFormat("%v(%v)", name, value);
}

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
