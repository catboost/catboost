#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/testing/gtest_extensions/assertions.h>

#include <library/cpp/yt/yson_string/convert.h>

#include <thread>

namespace NYT::NYson {
namespace {

////////////////////////////////////////////////////////////////////////////////

template <class T, class R = T, class U>
void Check(const U& value)
{
    auto str = ConvertToYsonString(static_cast<T>(value));
    auto anotherValue = ConvertFromYsonString<R>(str);
    EXPECT_EQ(static_cast<T>(value), anotherValue);
}

TEST(TConvertTest, Basic)
{
    Check<i8>(13);
    Check<i32>(13);
    Check<i64>(13);
    Check<i8>(-13);
    Check<i32>(-13);
    Check<i64>(-13);
    Check<ui8>(13);
    Check<ui32>(13);
    Check<ui64>(13);
    Check<TString>("");
    Check<TString>("hello");
    Check<TStringBuf, TString>("hello");
    Check<const char*, TString>("hello");
    Check<float>(3.14);
    Check<double>(3.14);
    Check<bool>(true);
    Check<bool>(false);
    Check<TInstant>(TInstant::Now());
    Check<TDuration>(TDuration::Seconds(123));
    Check<TGuid>(TGuid::FromString("12345678-12345678-abcdabcd-fefefefe"));
}

TEST(TConvertTest, InRange)
{
    EXPECT_EQ(ConvertFromYsonString<i16>(ConvertToYsonString(static_cast<i64>(-123))), -123);
    EXPECT_EQ(ConvertFromYsonString<ui16>(ConvertToYsonString(static_cast<ui64>(123))), 123U);
}

TEST(TConvertTest, OutOfRange)
{
    EXPECT_THROW_MESSAGE_HAS_SUBSTR(
        ConvertFromYsonString<i8>(ConvertToYsonString(static_cast<i64>(128))),
        TYsonLiteralParseException,
        "is out of expected range");
    EXPECT_THROW_MESSAGE_HAS_SUBSTR(
        ConvertFromYsonString<ui8>(ConvertToYsonString(static_cast<ui64>(256))),
        TYsonLiteralParseException,
        "is out of expected range");
}

TEST(TConvertTest, MalformedValues)
{
    EXPECT_THROW_MESSAGE_HAS_SUBSTR(
        ConvertFromYsonString<TInstant>(ConvertToYsonString(TStringBuf("sometime"))),
        TYsonLiteralParseException,
        "Error parsing \"instant\" value");
    EXPECT_THROW_MESSAGE_HAS_SUBSTR(
        ConvertFromYsonString<TGuid>(ConvertToYsonString(TStringBuf("1-2-3-g"))),
        TYsonLiteralParseException,
        "Error parsing \"guid\" value");
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT::NYson
