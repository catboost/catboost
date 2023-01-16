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
        /*initializeStorage*/ false);
    auto* ptr = buffer.Begin();
    *ptr++ = NDetail::StringMarker;
    ptr += WriteVarInt64(ptr, static_cast<i64>(value.length()));
    ::memcpy(ptr, value.data(), value.length());
    ptr += value.length();
    return TYsonString(buffer.Slice(buffer.Begin(), ptr));
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
        ythrow yexception() << "missing type marker";
    }
    if (ch != NDetail::Int64Marker) {
        ythrow yexception() << Format("unexpected %v",
            FormatUnexpectedMarker(ch));
    }
    i64 result;
    try {
        ReadVarInt64(&input, &result);
    } catch (const std::exception& ex) {
        ythrow TYsonLiteralParseException() << "failed to decode \"int64\" value: " <<
            ex.what();
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
        ythrow yexception() << "missing type marker";
    }
    if (ch != NDetail::Uint64Marker) {
        ythrow yexception() << Format("unexpected %v",
            FormatUnexpectedMarker(ch));
    }
    ui64 result;
    try {
        ReadVarUint64(&input, &result);
    } catch (const std::exception& ex) {
        ythrow TYsonLiteralParseException() << "failed to decode \"uint64\" value: " <<
            ex.what();
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
        ythrow yexception() << "missing type marker";
    }
    if (ch != NDetail::StringMarker) {
        ythrow yexception() << Format("unexpected %v",
            FormatUnexpectedMarker(ch));
    }
    i64 length;
    try {
        ReadVarInt64(&input, &length);
    } catch (const std::exception& ex) {
        ythrow yexception() << "failed to decode string length: " <<
            ex.what();
    }
    if (length < 0) {
        ythrow yexception() << "negative string length " <<
            length;
    }
    if (static_cast<i64>(input.Avail()) != length) {
        ythrow TYsonLiteralParseException() << Format("incorrect remaining string length: expected %v, got %v",
            length,
            input.Avail());
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
        ythrow yexception() << "missing type marker";
    }
    if (ch != NDetail::DoubleMarker) {
        ythrow yexception() << Format("unexpected %v",
            FormatUnexpectedMarker(ch));
    }
    if (input.Avail() != sizeof(double)) {
        ythrow TYsonLiteralParseException() << Format("incorrect remaining string length: expected %v, got %v",
            sizeof(double),
            input.Avail());
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
            ythrow TYsonLiteralParseException() << "Error parsing \"" #type "\" value from YSON: " << \
                ex.what(); \
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
        ythrow TYsonLiteralParseException() << "Error parsing \"string\" value from YSON: " <<
            ex.what();
    }
}

template <>
float ConvertFromYsonString<float>(const TYsonStringBuf& str)
{
    try {
        return static_cast<float>(ParseDoubleFromYsonString(str));
    } catch (const std::exception& ex) {
        ythrow TYsonLiteralParseException() << "Error parsing \"float\" value from YSON: " <<
            ex.what();
    }
}

template <>
double ConvertFromYsonString<double>(const TYsonStringBuf& str)
{
    try {
        return ParseDoubleFromYsonString(str);
    } catch (const std::exception& ex) {
        ythrow TYsonLiteralParseException() << "Error parsing \"double\" value from YSON: " <<
            ex.what();
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
            ythrow yexception() << "missing type marker";
        }
        if (ch != NDetail::TrueMarker && ch != NDetail::FalseMarker) {
            ythrow yexception() << Format("unexpected %v",
                FormatUnexpectedMarker(ch));
        }
        return ch == NDetail::TrueMarker;
    } catch (const std::exception& ex) {
        ythrow TYsonLiteralParseException() << "Error parsing \"boolean\" value from YSON: " <<
            ex.what();
    }
}

template <>
TInstant ConvertFromYsonString<TInstant>(const TYsonStringBuf& str)
{
    try {
        return TInstant::ParseIso8601(ParseStringFromYsonString(str));
    } catch (const std::exception& ex) {
        ythrow TYsonLiteralParseException() << "Error parsing \"instant\" value from YSON: " <<
            ex.what();
    }
}

template <>
TDuration ConvertFromYsonString<TDuration>(const TYsonStringBuf& str)
{
    try {
        return TDuration::MilliSeconds(ParseUint64FromYsonString(str));
    } catch (const std::exception& ex) {
        ythrow TYsonLiteralParseException() << "Error parsing \"duration\" value from YSON: " <<
            ex.what();
    }
}

template <>
TGuid ConvertFromYsonString<TGuid>(const TYsonStringBuf& str)
{
    try {
        return TGuid::FromString(ParseStringFromYsonString(str));
    } catch (const std::exception& ex) {
        ythrow TYsonLiteralParseException() << "Error parsing \"guid\" value from YSON: " <<
            ex.what();
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NYson
