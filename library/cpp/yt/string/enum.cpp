#include "enum.h"

#include "format.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

////////////////////////////////////////////////////////////////////////////////

#if defined(_MSC_VER)

extern "C" TEnumSuggestionsCalculator TryGetEnumSuggestionsCalculatorWeak()
{
    return nullptr;
}

__pragma(comment(linker, "/alternatename:TryGetEnumSuggestionsCalculator=TryGetEnumSuggestionsCalculatorWeak"))

#else

extern "C" Y_WEAK TEnumSuggestionsCalculator TryGetEnumSuggestionsCalculator()
{
    return nullptr;
}

#endif

////////////////////////////////////////////////////////////////////////////////

void ThrowMalformedEnumValueException(
    TStringBuf typeName,
    TStringBuf value,
    const std::span<const TStringBuf>& domainNames)
{
    auto errorMessage = Format("Error parsing %v value %Qv", typeName, value);
    auto suggestionsCalculator = TryGetEnumSuggestionsCalculator();
    if (!domainNames.empty() && suggestionsCalculator) {
        errorMessage += Format("; closest possible values are %v", suggestionsCalculator(value, domainNames));
    }
    throw TSimpleException(errorMessage);
}

template <bool ThrowOnError>
std::optional<std::string> DecodeEnumValueImpl(TStringBuf value)
{
    auto camelValue = UnderscoreCaseToCamelCase(value);
    auto underscoreValue = CamelCaseToUnderscoreCase(camelValue);
    if (value != underscoreValue) {
        if constexpr (ThrowOnError) {
            throw TSimpleException(Format("Enum value %Qv is not in a proper underscore case; did you mean %Qv?",
                value,
                underscoreValue));
        } else {
            return std::nullopt;
        }
    }
    return camelValue;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

std::optional<std::string> TryDecodeEnumValue(TStringBuf value)
{
    return NDetail::DecodeEnumValueImpl<false>(value);
}

std::string DecodeEnumValue(TStringBuf value)
{
    auto decodedValue = NDetail::DecodeEnumValueImpl<true>(value);
    YT_VERIFY(decodedValue);
    return *decodedValue;
}

std::string EncodeEnumValue(TStringBuf value)
{
    return CamelCaseToUnderscoreCase(value);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
