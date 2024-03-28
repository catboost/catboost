#include "string.h"
#include "format.h"

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/exception/exception.h>

#include <util/generic/hash.h>

#include <util/string/ascii.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

void UnderscoreCaseToCamelCase(TStringBuilderBase* builder, TStringBuf str)
{
    bool first = true;
    bool upper = true;
    for (char c : str) {
        if (c == '_') {
            upper = true;
        } else {
            if (upper) {
                if (!std::isalpha(c) && !first) {
                    builder->AppendChar('_');
                }
                c = std::toupper(c);
            }
            builder->AppendChar(c);
            upper = false;
        }
        first = false;
    }
}

TString UnderscoreCaseToCamelCase(TStringBuf str)
{
    TStringBuilder builder;
    UnderscoreCaseToCamelCase(&builder, str);
    return builder.Flush();
}

void CamelCaseToUnderscoreCase(TStringBuilderBase* builder, TStringBuf str)
{
    bool first = true;
    for (char c : str) {
        if (std::isupper(c) && std::isalpha(c)) {
            if (!first) {
                builder->AppendChar('_');
            }
            c = std::tolower(c);
        }
        builder->AppendChar(c);
        first = false;
    }
}

TString CamelCaseToUnderscoreCase(TStringBuf str)
{
    TStringBuilder builder;
    CamelCaseToUnderscoreCase(&builder, str);
    return builder.Flush();
}

////////////////////////////////////////////////////////////////////////////////

TString TrimLeadingWhitespaces(const TString& str)
{
    for (int i = 0; i < static_cast<int>(str.size()); ++i) {
        if (str[i] != ' ') {
            return str.substr(i);
        }
    }
    return "";
}

TString Trim(const TString& str, const TString& whitespaces)
{
    size_t end = str.size();
    while (end > 0) {
        size_t i = end - 1;
        bool isWhitespace = false;
        for (auto c : whitespaces) {
            if (str[i] == c) {
                isWhitespace = true;
                break;
            }
        }
        if (!isWhitespace) {
            break;
        }
        --end;
    }

    if (end == 0) {
        return "";
    }

    size_t begin = str.find_first_not_of(whitespaces);
    YT_VERIFY(begin != TString::npos);
    YT_VERIFY(begin < end);
    return str.substr(begin, end - begin);
}

////////////////////////////////////////////////////////////////////////////////

namespace {

const ui16 DecimalDigits2[100] = {
    12336,  12592,  12848,  13104,  13360,  13616,  13872,  14128,  14384,  14640,
    12337,  12593,  12849,  13105,  13361,  13617,  13873,  14129,  14385,  14641,
    12338,  12594,  12850,  13106,  13362,  13618,  13874,  14130,  14386,  14642,
    12339,  12595,  12851,  13107,  13363,  13619,  13875,  14131,  14387,  14643,
    12340,  12596,  12852,  13108,  13364,  13620,  13876,  14132,  14388,  14644,
    12341,  12597,  12853,  13109,  13365,  13621,  13877,  14133,  14389,  14645,
    12342,  12598,  12854,  13110,  13366,  13622,  13878,  14134,  14390,  14646,
    12343,  12599,  12855,  13111,  13367,  13623,  13879,  14135,  14391,  14647,
    12344,  12600,  12856,  13112,  13368,  13624,  13880,  14136,  14392,  14648,
    12345,  12601,  12857,  13113,  13369,  13625,  13881,  14137,  14393,  14649
};

template <class T>
char* WriteSignedDecIntToBufferBackwardsImpl(char* ptr, T value, TStringBuf min)
{
    if (value == 0) {
        --ptr;
        *ptr = '0';
        return ptr;
    }

    // The negative value handling code below works incorrectly for min values.
    if (value == std::numeric_limits<T>::min()) {
        ptr -= min.length();
        ::memcpy(ptr, min.begin(), min.length());
        return ptr;
    }

    bool negative = false;
    if (value < 0) {
        negative = true;
        value = -value;
    }

    while (value >= 10) {
        auto rem = value % 100;
        auto quot = value / 100;
        ptr -= 2;
        ::memcpy(ptr, &DecimalDigits2[rem], 2);
        value = quot;
    }

    if (value > 0) {
        --ptr;
        *ptr = ('0' + value);
    }

    if (negative) {
        --ptr;
        *ptr = '-';
    }

    return ptr;
}

template <class T>
char* WriteUnsignedDecIntToBufferBackwardsImpl(char* ptr, T value)
{
    if (value == 0) {
        --ptr;
        *ptr = '0';
        return ptr;
    }

    while (value >= 10) {
        auto rem = value % 100;
        auto quot = value / 100;
        ptr -= 2;
        ::memcpy(ptr, &DecimalDigits2[rem], 2);
        value = quot;
    }

    if (value > 0) {
        --ptr;
        *ptr = ('0' + value);
    }

    return ptr;
}

} // namespace

template <>
char* WriteDecIntToBufferBackwards(char* ptr, i32 value)
{
    return WriteSignedDecIntToBufferBackwardsImpl(ptr, value, TStringBuf("-2147483647"));
}

template <>
char* WriteDecIntToBufferBackwards(char* ptr, i64 value)
{
    return WriteSignedDecIntToBufferBackwardsImpl(ptr, value, TStringBuf("-9223372036854775808"));
}

template <>
char* WriteDecIntToBufferBackwards(char* ptr, ui32 value)
{
    return WriteUnsignedDecIntToBufferBackwardsImpl(ptr, value);
}

template <>
char* WriteDecIntToBufferBackwards(char* ptr, ui64 value)
{
    return WriteUnsignedDecIntToBufferBackwardsImpl(ptr, value);
}

////////////////////////////////////////////////////////////////////////////////

namespace {

template <class T>
char* WriteSignedHexIntToBufferBackwardsImpl(char* ptr, T value, bool uppercase, TStringBuf min)
{
    if (value == 0) {
        --ptr;
        *ptr = '0';
        return ptr;
    }

    // The negative value handling code below works incorrectly for min values.
    if (value == std::numeric_limits<T>::min()) {
        ptr -= min.length();
        ::memcpy(ptr, min.begin(), min.length());
        return ptr;
    }

    bool negative = false;
    if (value < 0) {
        negative = true;
        value = -value;
    }

    while (value != 0) {
        auto rem = value & 0xf;
        auto quot = value >> 4;
        --ptr;
        *ptr = uppercase ? IntToHexUppercase[rem] : IntToHexLowercase[rem];
        value = quot;
    }

    if (negative) {
        --ptr;
        *ptr = '-';
    }

    return ptr;
}

template <class T>
char* WriteUnsignedHexIntToBufferBackwardsImpl(char* ptr, T value, bool uppercase)
{
    if (value == 0) {
        --ptr;
        *ptr = '0';
        return ptr;
    }

    while (value != 0) {
        auto rem = value & 0xf;
        auto quot = value >> 4;
        --ptr;
        *ptr = uppercase ? IntToHexUppercase[rem] : IntToHexLowercase[rem];
        value = quot;
    }

    return ptr;
}

} // namespace

template <>
char* WriteHexIntToBufferBackwards(char* ptr, i32 value, bool uppercase)
{
    return WriteSignedHexIntToBufferBackwardsImpl(ptr, value, uppercase, TStringBuf("-80000000"));
}

template <>
char* WriteHexIntToBufferBackwards(char* ptr, i64 value, bool uppercase)
{
    return WriteSignedHexIntToBufferBackwardsImpl(ptr, value, uppercase, TStringBuf("-8000000000000000"));
}

template <>
char* WriteHexIntToBufferBackwards(char* ptr, ui32 value, bool uppercase)
{
    return WriteUnsignedHexIntToBufferBackwardsImpl(ptr, value, uppercase);
}

template <>
char* WriteHexIntToBufferBackwards(char* ptr, ui64 value, bool uppercase)
{
    return WriteUnsignedHexIntToBufferBackwardsImpl(ptr, value, uppercase);
}

////////////////////////////////////////////////////////////////////////////////

size_t TCaseInsensitiveStringHasher::operator()(TStringBuf arg) const
{
    auto compute = [&] (char* buffer) {
        for (size_t index = 0; index < arg.length(); ++index) {
            buffer[index] = AsciiToLower(arg[index]);
        }
        return ComputeHash(TStringBuf(buffer, arg.length()));
    };
    const size_t SmallSize = 256;
    if (arg.length() <= SmallSize) {
        std::array<char, SmallSize> stackBuffer;
        return compute(stackBuffer.data());
    } else {
        std::unique_ptr<char[]> heapBuffer(new char[arg.length()]);
        return compute(heapBuffer.get());
    }
}

bool TCaseInsensitiveStringEqualityComparer::operator()(TStringBuf lhs, TStringBuf rhs) const
{
    return AsciiEqualsIgnoreCase(lhs, rhs);
}

////////////////////////////////////////////////////////////////////////////////

bool TryParseBool(TStringBuf value, bool* result)
{
    if (value == "true" || value == "1") {
        *result = true;
        return true;
    } else if (value == "false" || value == "0") {
        *result = false;
        return true;
    } else {
        return false;
    }
}

bool ParseBool(TStringBuf value)
{
    bool result;
    if (!TryParseBool(value, &result)) {
        throw TSimpleException(Format("Error parsing boolean value %Qv",
            value));
    }
    return result;
}

TStringBuf FormatBool(bool value)
{
    return value ? TStringBuf("true") : TStringBuf("false");
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
