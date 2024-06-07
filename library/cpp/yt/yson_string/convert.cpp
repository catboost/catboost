#include "convert.h"
#include "format.h"

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/string/format.h>

#include <library/cpp/yt/coding/varint.h>

#include <library/cpp/yt/misc/cast.h>

#include <array>

#include <util/stream/mem.h>

namespace NYT::NYson {

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

} // namespace NYT::NYson
