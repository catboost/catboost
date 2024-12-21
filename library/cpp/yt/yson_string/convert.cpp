#include "convert.h"
#include "format.h"

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/string/format.h>

#include <library/cpp/yt/coding/varint.h>

#include <library/cpp/yt/misc/cast.h>

#include <array>

#include <util/string/escape.h>

#include <util/stream/mem.h>

namespace NYT::NYson {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

size_t FloatToStringWithNanInf(double value, char* buf, size_t size)
{
    if (std::isfinite(value)) {
        return FloatToString(value, buf, size);
    }

    static const TStringBuf nanLiteral = "%nan";
    static const TStringBuf infLiteral = "%inf";
    static const TStringBuf negativeInfLiteral = "%-inf";

    TStringBuf str;
    if (std::isnan(value)) {
        str = nanLiteral;
    } else if (std::isinf(value) && value > 0) {
        str = infLiteral;
    } else {
        str = negativeInfLiteral;
    }
    YT_VERIFY(str.size() + 1 <= size);
    ::memcpy(buf, str.data(), str.size() + 1);
    return str.size();
}

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

template <>
TYsonString ConvertToYsonString<i8>(const i8& value)
{
    return ConvertToYsonString(static_cast<i64>(value));
}

template <>
TYsonString ConvertToYsonString<i32>(const i32& value)
{
    return ConvertToYsonString(static_cast<i64>(value));
}

template <>
TYsonString ConvertToYsonString<i64>(const i64& value)
{
    std::array<char, 1 + MaxVarInt64Size> buffer;
    auto* ptr = buffer.data();
    *ptr++ = NDetail::Int64Marker;
    ptr += WriteVarInt64(ptr, value);
    return TYsonString(TStringBuf(buffer.data(), ptr - buffer.data()));
}

template <>
TYsonString ConvertToYsonString<ui8>(const ui8& value)
{
    return ConvertToYsonString(static_cast<ui64>(value));
}

template <>
TYsonString ConvertToYsonString<ui32>(const ui32& value)
{
    return ConvertToYsonString(static_cast<ui64>(value));
}

template <>
TYsonString ConvertToYsonString<ui64>(const ui64& value)
{
    std::array<char, 1 + MaxVarInt64Size> buffer;
    auto* ptr = buffer.data();
    *ptr++ = NDetail::Uint64Marker;
    ptr += WriteVarUint64(ptr, value);
    return TYsonString(TStringBuf(buffer.data(), ptr - buffer.data()));
}

template <>
TYsonString ConvertToYsonString<TString>(const TString& value)
{
    return ConvertToYsonString(static_cast<TStringBuf>(value));
}

template <>
TYsonString ConvertToYsonString<std::string>(const std::string& value)
{
    return ConvertToYsonString(static_cast<TStringBuf>(value));
}

struct TConvertStringToYsonStringTag
{ };

template <>
TYsonString ConvertToYsonString<TStringBuf>(const TStringBuf& value)
{
    auto buffer = TSharedMutableRef::Allocate<TConvertStringToYsonStringTag>(
        1 + MaxVarInt64Size + value.length(),
        {.InitializeStorage = false});
    auto* ptr = buffer.Begin();
    *ptr++ = NDetail::StringMarker;
    ptr += WriteVarInt64(ptr, static_cast<i64>(value.length()));
    ::memcpy(ptr, value.data(), value.length());
    ptr += value.length();
    return TYsonString(buffer.Slice(buffer.Begin(), ptr));
}

TYsonString ConvertToYsonString(const char* value)
{
    return ConvertToYsonString(TStringBuf(value));
}

template <>
TYsonString ConvertToYsonString<float>(const float& value)
{
    return ConvertToYsonString(static_cast<double>(value));
}

template <>
TYsonString ConvertToYsonString<double>(const double& value)
{
    std::array<char, 1 + sizeof(double)> buffer;
    auto* ptr = buffer.data();
    *ptr++ = NDetail::DoubleMarker;
    ::memcpy(ptr, &value, sizeof(value));
    ptr += sizeof(value);
    return TYsonString(TStringBuf(buffer.data(), ptr - buffer.data()));
}

template <>
TYsonString ConvertToYsonString<bool>(const bool& value)
{
    char ch = value ? NDetail::TrueMarker : NDetail::FalseMarker;
    return TYsonString(TStringBuf(&ch, 1));
}

template <>
TYsonString ConvertToYsonString<TInstant>(const TInstant& value)
{
    return ConvertToYsonString(value.ToString());
}

template <>
TYsonString ConvertToYsonString<TDuration>(const TDuration& value)
{
    return ConvertToYsonString(value.MilliSeconds());
}

template <>
TYsonString ConvertToYsonString<TGuid>(const TGuid& value)
{
    std::array<char, MaxGuidStringSize> guidBuffer;
    auto guidLength = WriteGuidToBuffer(guidBuffer.data(), value) - guidBuffer.data();
    std::array<char, 1 + MaxVarInt64Size + MaxGuidStringSize> ysonBuffer;
    auto* ptr = ysonBuffer.data();
    *ptr++ = NDetail::StringMarker;
    ptr += WriteVarInt64(ptr, static_cast<i64>(guidLength));
    ::memcpy(ptr, guidBuffer.data(), guidLength);
    ptr += guidLength;
    return TYsonString(TStringBuf(ysonBuffer.data(), ptr - ysonBuffer.data()));
}

////////////////////////////////////////////////////////////////////////////////

namespace {

TString FormatUnexpectedMarker(char ch)
{
    switch (ch) {
        case NDetail::BeginListSymbol:
            return "list";
        case NDetail::BeginMapSymbol:
            return "map";
        case NDetail::BeginAttributesSymbol:
            return "attributes";
        case NDetail::EntitySymbol:
            return "\"entity\" literal";
        case NDetail::StringMarker:
            return "\"string\" literal";
        case NDetail::Int64Marker:
            return "\"int64\" literal";
        case NDetail::DoubleMarker:
            return "\"double\" literal";
        case NDetail::FalseMarker:
        case NDetail::TrueMarker:
            return "\"boolean\" literal";
        case NDetail::Uint64Marker:
            return "\"uint64\" literal";
        default:
            return Format("unexpected symbol %qv", ch);
    }
}

i64 ParseInt64FromYsonString(const TYsonStringBuf& str)
{
    YT_ASSERT(str.GetType() == EYsonType::Node);
    auto strBuf = str.AsStringBuf();
    TMemoryInput input(strBuf.data(), strBuf.length());
    char ch;
    if (!input.ReadChar(ch)) {
        throw TYsonLiteralParseException("Missing type marker");
    }
    if (ch != NDetail::Int64Marker) {
        throw TYsonLiteralParseException(Format("Unexpected %v",
            FormatUnexpectedMarker(ch)));
    }
    i64 result;
    try {
        ReadVarInt64(&input, &result);
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Failed to decode \"int64\" value");
    }
    return result;
}

ui64 ParseUint64FromYsonString(const TYsonStringBuf& str)
{
    YT_ASSERT(str.GetType() == EYsonType::Node);
    auto strBuf = str.AsStringBuf();
    TMemoryInput input(strBuf.data(), strBuf.length());
    char ch;
    if (!input.ReadChar(ch)) {
        throw TYsonLiteralParseException("Missing type marker");
    }
    if (ch != NDetail::Uint64Marker) {
        throw TYsonLiteralParseException(Format("Unexpected %v",
            FormatUnexpectedMarker(ch)));
    }
    ui64 result;
    try {
        ReadVarUint64(&input, &result);
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Failed to decode \"uint64\" value");
    }
    return result;
}

TString ParseStringFromYsonString(const TYsonStringBuf& str)
{
    YT_ASSERT(str.GetType() == EYsonType::Node);
    auto strBuf = str.AsStringBuf();
    TMemoryInput input(strBuf.data(), strBuf.length());
    char ch;
    if (!input.ReadChar(ch)) {
        throw TYsonLiteralParseException("Missing type marker");
    }
    if (ch != NDetail::StringMarker) {
        throw TYsonLiteralParseException(Format("Unexpected %v",
            FormatUnexpectedMarker(ch)));
    }
    i64 length;
    try {
        ReadVarInt64(&input, &length);
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Failed to decode string length");
    }
    if (length < 0) {
        throw TYsonLiteralParseException(Format(
            "Negative string length %v",
            length));
    }
    if (static_cast<i64>(input.Avail()) != length) {
        throw TYsonLiteralParseException(Format("Incorrect remaining string length: expected %v, got %v",
            length,
            input.Avail()));
    }
    TString result;
    result.ReserveAndResize(length);
    YT_VERIFY(static_cast<i64>(input.Read(result.Detach(), length)) == length);
    return result;
}

double ParseDoubleFromYsonString(const TYsonStringBuf& str)
{
    YT_ASSERT(str.GetType() == EYsonType::Node);
    auto strBuf = str.AsStringBuf();
    TMemoryInput input(strBuf.data(), strBuf.length());
    char ch;
    if (!input.ReadChar(ch)) {
        throw TYsonLiteralParseException("Missing type marker");
    }
    if (ch != NDetail::DoubleMarker) {
        throw TYsonLiteralParseException(Format("Unexpected %v",
            FormatUnexpectedMarker(ch)));
    }
    if (input.Avail() != sizeof(double)) {
        throw TYsonLiteralParseException(Format("Incorrect remaining string length: expected %v, got %v",
            sizeof(double),
            input.Avail()));
    }
    double result;
    YT_VERIFY(input.Read(&result, sizeof(result)));
    return result;
}

} // namespace

#define PARSE(type, underlyingType) \
    template <> \
    type ConvertFromYsonString<type>(const TYsonStringBuf& str) \
    { \
        try { \
            return CheckedIntegralCast<type>(Parse ## underlyingType ## FromYsonString(str)); \
        } catch (const std::exception& ex) { \
            throw TYsonLiteralParseException(ex, "Error parsing \"" #type "\" value from YSON"); \
        } \
    }

PARSE(i8,   Int64 )
PARSE(i16,  Int64 )
PARSE(i32,  Int64 )
PARSE(i64,  Int64 )
PARSE(ui8,  Uint64)
PARSE(ui16, Uint64)
PARSE(ui32, Uint64)
PARSE(ui64, Uint64)

#undef PARSE

template <>
TString ConvertFromYsonString<TString>(const TYsonStringBuf& str)
{
    try {
        return ParseStringFromYsonString(str);
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"string\" value from YSON");
    }
}

template <>
float ConvertFromYsonString<float>(const TYsonStringBuf& str)
{
    try {
        return static_cast<float>(ParseDoubleFromYsonString(str));
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"float\" value from YSON");
    }
}

template <>
double ConvertFromYsonString<double>(const TYsonStringBuf& str)
{
    try {
        return ParseDoubleFromYsonString(str);
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"double\" value from YSON");
    }
}

template <>
bool ConvertFromYsonString<bool>(const TYsonStringBuf& str)
{
    try {
        YT_ASSERT(str.GetType() == EYsonType::Node);
        auto strBuf = str.AsStringBuf();
        TMemoryInput input(strBuf.data(), strBuf.length());
        char ch;
        if (!input.ReadChar(ch)) {
            throw TYsonLiteralParseException("Missing type marker");
        }
        if (ch != NDetail::TrueMarker && ch != NDetail::FalseMarker) {
            throw TYsonLiteralParseException(Format("Unexpected %v",
                FormatUnexpectedMarker(ch)));
        }
        return ch == NDetail::TrueMarker;
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"boolean\" value from YSON");
    }
}

template <>
TInstant ConvertFromYsonString<TInstant>(const TYsonStringBuf& str)
{
    try {
        return TInstant::ParseIso8601(ParseStringFromYsonString(str));
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"instant\" value from YSON");
    }
}

template <>
TDuration ConvertFromYsonString<TDuration>(const TYsonStringBuf& str)
{
    try {
        return TDuration::MilliSeconds(ParseUint64FromYsonString(str));
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"duration\" value from YSON");
    }
}

template <>
TGuid ConvertFromYsonString<TGuid>(const TYsonStringBuf& str)
{
    try {
        return TGuid::FromString(ParseStringFromYsonString(str));
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"guid\" value from YSON");
    }
}

////////////////////////////////////////////////////////////////////////////////

template <>
TYsonString ConvertToTextYsonString<i8>(const i8& value)
{
    return ConvertToTextYsonString(static_cast<i64>(value));
}

template <>
TYsonString ConvertToTextYsonString<i32>(const i32& value)
{
    return ConvertToTextYsonString(static_cast<i64>(value));
}

template <>
TYsonString ConvertToTextYsonString<i64>(const i64& value)
{
    return TYsonString{::ToString(value)};
}

template <>
TYsonString ConvertToTextYsonString<ui8>(const ui8& value)
{
    return ConvertToTextYsonString(static_cast<ui64>(value));
}

template <>
TYsonString ConvertToTextYsonString<ui32>(const ui32& value)
{
    return ConvertToTextYsonString(static_cast<ui64>(value));
}

template <>
TYsonString ConvertToTextYsonString<ui64>(const ui64& value)
{
    return TYsonString{::ToString(value) + 'u'};
}

template <>
TYsonString ConvertToTextYsonString<TString>(const TString& value)
{
    return ConvertToTextYsonString(TStringBuf(value));
}

template <>
TYsonString ConvertToTextYsonString<std::string>(const std::string& value)
{
    return ConvertToTextYsonString(TStringBuf(value));
}

template <>
TYsonString ConvertToTextYsonString<TStringBuf>(const TStringBuf& value)
{
    return TYsonString(NYT::Format("\"%v\"", ::EscapeC(value)));
}

template <>
TYsonString ConvertToTextYsonString<std::string_view>(const std::string_view& value)
{
    return ConvertToTextYsonString(TStringBuf(value));
}

TYsonString ConvertToTextYsonString(const char* value)
{
    return ConvertToTextYsonString(TStringBuf(value));
}

template <>
TYsonString ConvertToTextYsonString<float>(const float& value)
{
    return ConvertToTextYsonString(static_cast<double>(value));
}

template <>
TYsonString ConvertToTextYsonString<double>(const double& value)
{
    char buf[256];
    auto str = TStringBuf(buf, NDetail::FloatToStringWithNanInf(value, buf, sizeof(buf)));
    auto ret = NYT::Format(
        "%v%v",
        str,
        MakeFormatterWrapper([&] (TStringBuilderBase* builder) {
            if (str.find('.') == TString::npos && str.find('e') == TString::npos && std::isfinite(value)) {
                builder->AppendChar('.');
            }
        }));
    return TYsonString(std::move(ret));
}

template <>
TYsonString ConvertToTextYsonString<bool>(const bool& value)
{
    return value
        ? TYsonString(TStringBuf("%true"))
        : TYsonString(TStringBuf("%false"));
}

template <>
TYsonString ConvertToTextYsonString<TInstant>(const TInstant& value)
{
    return ConvertToTextYsonString(value.ToString());
}

template <>
TYsonString ConvertToTextYsonString<TDuration>(const TDuration& value)
{
    // ConvertTo does unchecked cast to i64 :(.
    return ConvertToTextYsonString(static_cast<i64>(value.MilliSeconds()));
}

template <>
TYsonString ConvertToTextYsonString<TGuid>(const TGuid& value)
{
    return ConvertToTextYsonString(NYT::ToString(value));
}

////////////////////////////////////////////////////////////////////////////////

namespace {

template <class TSomeInt>
TSomeInt ReadTextUint(TStringBuf strBuf)
{
    // Drop 'u'
    return ::FromString<TSomeInt>(TStringBuf{strBuf.data(), strBuf.length() - 1});
}

template <class TSomeInt>
TSomeInt ReadTextInt(TStringBuf strBuf)
{
    return ::FromString<TSomeInt>(TStringBuf{strBuf.data(), strBuf.length()});
}

bool IsNumeric(TStringBuf strBuf)
{
    bool isNumeric = true;
    bool isNegative = false;
    for (int i = 0; i < std::ssize(strBuf); ++i) {
        char c = strBuf[i];

        if (!('0' <= c && c <= '9')) {
            if (i == 0 && c == '-') {
                isNegative = true;
                continue;
            }
            if (i == std::ssize(strBuf) - 1 && c == 'u' && !isNegative) {
                continue;
            }
            isNumeric = false;
            break;
        }
    }

    return isNumeric;
}

////////////////////////////////////////////////////////////////////////////////

template <class TSomeInt>
TSomeInt ParseSomeIntFromTextYsonString(const TYsonStringBuf& str)
{
    YT_ASSERT(str.GetType() == EYsonType::Node);
    auto strBuf = str.AsStringBuf();

    if (std::ssize(strBuf) == 0 || !IsNumeric(strBuf)) {
        throw TYsonLiteralParseException(NYT::Format(
            "Unexpected %v\n"
            "Value is not numeric",
            strBuf));
    }

    if (strBuf.back() == 'u') {
        // Drop 'u'
        return ReadTextUint<TSomeInt>(strBuf);
    } else {
        return ReadTextInt<TSomeInt>(strBuf);
    }
}

////////////////////////////////////////////////////////////////////////////////

TString DoParseStringFromTextYson(TStringBuf strBuf)
{
    // Remove quotation marks.
    return ::UnescapeC(TStringBuf{strBuf.data() + 1, strBuf.length() - 2});
}

TString ParseStringFromTextYsonString(const TYsonStringBuf& str)
{
    YT_ASSERT(str.GetType() == EYsonType::Node);
    auto strBuf = str.AsStringBuf();
    if (std::ssize(strBuf) < 2 || strBuf.front() != '\"' || strBuf.back() != '\"') {
        throw TYsonLiteralParseException(Format(
            "Unexpected %v\n"
            "Text yson string must begin and end with \\\"",
            strBuf));
    }
    return DoParseStringFromTextYson(strBuf);
}

////////////////////////////////////////////////////////////////////////////////

double ParseDoubleFromTextYsonString(const TYsonStringBuf& str)
{
    YT_ASSERT(str.GetType() == EYsonType::Node);
    auto strBuf = str.AsStringBuf();

    if (std::ssize(strBuf) < 2) {
        throw TYsonLiteralParseException(Format(
            "Incorrect remaining string length: expected at least 2, got %v",
            std::ssize(strBuf)));
    }

    // Check special values first.
    // %nan
    // %inf, %+inf, %-inf
    if (strBuf[0] == '%') {
        switch (strBuf[1]) {
            case '+':
            case 'i':
                return std::numeric_limits<double>::infinity();

            case '-':
                return -std::numeric_limits<double>::infinity();

            case 'n':
                return std::numeric_limits<double>::quiet_NaN();

            default:
                throw TYsonLiteralParseException(Format(
                    "Incorrect %%-literal %v",
                    strBuf));
        }
    }

    return ::FromString<double>(strBuf);
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

#define PARSE_INT(type, underlyingType) \
    template <> \
    type ConvertFromTextYsonString<type>(const TYsonStringBuf& str) \
    { \
        try { \
            return CheckedIntegralCast<type>(ParseSomeIntFromTextYsonString<underlyingType>(str)); \
        } catch (const std::exception& ex) { \
            throw TYsonLiteralParseException(ex, "Error parsing \"" #type "\" value from YSON"); \
        } \
    }

PARSE_INT(i8,    i64)
PARSE_INT(i16,   i64)
PARSE_INT(i32,   i64)
PARSE_INT(i64,   i64)
PARSE_INT(ui8,  ui64)
PARSE_INT(ui16, ui64)
PARSE_INT(ui32, ui64)
PARSE_INT(ui64, ui64)

#undef PARSE

template <>
TString ConvertFromTextYsonString<TString>(const TYsonStringBuf& str)
{
    try {
        return ParseStringFromTextYsonString(str);
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"string\" value from YSON");
    }
}

template <>
std::string ConvertFromTextYsonString<std::string>(const TYsonStringBuf& str)
{
    return std::string{ConvertFromTextYsonString<TString>(str)};
}

template <>
float ConvertFromTextYsonString<float>(const TYsonStringBuf& str)
{
    try {
        return static_cast<float>(ParseDoubleFromTextYsonString(str));
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"float\" value from YSON");
    }
}

template <>
double ConvertFromTextYsonString<double>(const TYsonStringBuf& str)
{
    try {
        return ParseDoubleFromTextYsonString(str);
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"double\" value from YSON");
    }
}

template <>
bool ConvertFromTextYsonString<bool>(const TYsonStringBuf& str)
{
    try {
        YT_ASSERT(str.GetType() == EYsonType::Node);
        auto strBuf = str.AsStringBuf();

        if (std::ssize(strBuf) == 0) {
            throw TYsonLiteralParseException("Empty string");
        }

        char ch = strBuf.front();

        if (ch == '%') {
            if (strBuf != "%true" && strBuf != "%false") {
                throw TYsonLiteralParseException(Format(
                    "Expected %%true or %%false but found %v",
                    strBuf));
            }
            return strBuf == "%true";
        }

        if (ch == '\"') {
            return ParseBool(DoParseStringFromTextYson(strBuf));
        }

        // NB(arkady-e1ppa): This check is linear in size(strBuf)
        // And thus is tried as the last resort.
        if (IsNumeric(strBuf)) {
            auto checkValue = [&] (const auto& functor) {
                auto value = functor(strBuf);
                if (value != 0 && value != 1) {
                    throw TYsonLiteralParseException(Format(
                        "Expected 0 or 1 but found %v",
                        value));
                }
                return static_cast<bool>(value);
            };

            if (strBuf.back() == 'u') {
                return checkValue(&ReadTextUint<ui64>);
            } else {
                return checkValue(&ReadTextInt<i64>);
            }
        }

        throw TYsonLiteralParseException(Format(
            "Unexpected %v\n"
            "No known conversion to \"boolean\" value",
            strBuf));
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"boolean\" value from YSON");
    }
}

template <>
TInstant ConvertFromTextYsonString<TInstant>(const TYsonStringBuf& str)
{
    try {
        return TInstant::ParseIso8601(ParseStringFromTextYsonString(str));
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"instant\" value from YSON");
    }
}

template <>
TDuration ConvertFromTextYsonString<TDuration>(const TYsonStringBuf& str)
{
    try {
        return TDuration::MilliSeconds(ParseSomeIntFromTextYsonString<i64>(str));
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"duration\" value from YSON");
    }
}

template <>
TGuid ConvertFromTextYsonString<TGuid>(const TYsonStringBuf& str)
{
    try {
        return TGuid::FromString(ParseStringFromTextYsonString(str));
    } catch (const std::exception& ex) {
        throw TYsonLiteralParseException(ex, "Error parsing \"guid\" value from YSON");
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NYson
