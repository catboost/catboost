#include "enum.h"
#include "format.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

TString DecodeEnumValue(TStringBuf value)
{
    auto camelValue = UnderscoreCaseToCamelCase(value);
    auto underscoreValue = CamelCaseToUnderscoreCase(camelValue);
    if (value != underscoreValue) {
        ythrow yexception() << Format("Enum value %Qv is not in a proper underscore case; did you mean %Qv?",
            value,
            underscoreValue);
    }
    return camelValue;
}

TString EncodeEnumValue(TStringBuf value)
{
    return CamelCaseToUnderscoreCase(value);
}

void ThrowEnumParsingError(TStringBuf name, TStringBuf value)
{
    ythrow yexception() << Format("Error parsing %v value %qv", name, value);
}

void FormatUnknownEnum(TStringBuilderBase* builder, TStringBuf name, i64 value)
{
    builder->AppendFormat("%v(%v)", name, value);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
