#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/string/format.h>

#include <library/cpp/yt/small_containers/compact_vector.h>

#include <util/generic/hash_set.h>

#include <limits>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

// Some compile-time sanity checks.
static_assert(CFormattable<int>);
static_assert(CFormattable<double>);
static_assert(CFormattable<void*>);
static_assert(CFormattable<const char*>);
static_assert(CFormattable<TStringBuf>);
static_assert(CFormattable<TString>);
static_assert(CFormattable<std::span<int>>);
static_assert(CFormattable<std::vector<int>>);

// N.B. TCompactVector<int, 1> is not buildable on Windows
static_assert(CFormattable<TCompactVector<int, 2>>);
static_assert(CFormattable<std::set<int>>);
static_assert(CFormattable<std::map<int, int>>);
static_assert(CFormattable<std::multimap<int, int>>);
static_assert(CFormattable<THashSet<int>>);
static_assert(CFormattable<THashMap<int, int>>);
static_assert(CFormattable<THashMultiSet<int>>);
static_assert(CFormattable<std::pair<int, int>>);
static_assert(CFormattable<std::optional<int>>);
static_assert(CFormattable<TDuration>);
static_assert(CFormattable<TInstant>);

struct TUnformattable
{ };
static_assert(!CFormattable<TUnformattable>);

////////////////////////////////////////////////////////////////////////////////

TEST(TFormatTest, Nothing)
{
    EXPECT_EQ("abc", Format("a%nb%nc", 1, 2));
}

TEST(TFormatTest, Verbatim)
{
    EXPECT_EQ("", Format(""));
    EXPECT_EQ("test", Format("test"));
    EXPECT_EQ("%", Format("%%"));
    EXPECT_EQ("%hello%world%", Format("%%hello%%world%%"));
}

TEST(TFormatTest, MultipleArgs)
{
    EXPECT_EQ("2+2=4", Format("%v+%v=%v", 2, 2, 4));
}

TEST(TFormatTest, Strings)
{
    EXPECT_EQ("test", Format("%s", "test"));
    EXPECT_EQ("test", Format("%s", TStringBuf("test")));
    EXPECT_EQ("test", Format("%s", TString("test")));

    EXPECT_EQ("   abc", Format("%6s", TString("abc")));
    EXPECT_EQ("abc   ", Format("%-6s", TString("abc")));
    EXPECT_EQ("       abc", Format("%10v", TString("abc")));
    EXPECT_EQ("abc       ", Format("%-10v", TString("abc")));
    EXPECT_EQ("abc", Format("%2s", TString("abc")));
    EXPECT_EQ("abc", Format("%-2s", TString("abc")));
    EXPECT_EQ("abc", Format("%0s", TString("abc")));
    EXPECT_EQ("abc", Format("%-0s", TString("abc")));
    EXPECT_EQ(100, std::ssize(Format("%100v", "abc")));
}

TEST(TFormatTest, DecIntegers)
{
    EXPECT_EQ("123", Format("%d", 123));
    EXPECT_EQ("123", Format("%v", 123));

    EXPECT_EQ("042", Format("%03d", 42));
    EXPECT_EQ("42", Format("%01d", 42));

    EXPECT_EQ("2147483647", Format("%d", std::numeric_limits<i32>::max()));
    EXPECT_EQ("-2147483648", Format("%d", std::numeric_limits<i32>::min()));

    EXPECT_EQ("0", Format("%u", 0U));
    EXPECT_EQ("0", Format("%v", 0U));
    EXPECT_EQ("4294967295", Format("%u", std::numeric_limits<ui32>::max()));
    EXPECT_EQ("4294967295", Format("%v", std::numeric_limits<ui32>::max()));

    EXPECT_EQ("9223372036854775807", Format("%" PRId64, std::numeric_limits<i64>::max()));
    EXPECT_EQ("9223372036854775807", Format("%v", std::numeric_limits<i64>::max()));
    EXPECT_EQ("-9223372036854775808", Format("%" PRId64, std::numeric_limits<i64>::min()));
    EXPECT_EQ("-9223372036854775808", Format("%v", std::numeric_limits<i64>::min()));

    EXPECT_EQ("0", Format("%" PRIu64, 0ULL));
    EXPECT_EQ("0", Format("%v", 0ULL));
    EXPECT_EQ("18446744073709551615", Format("%" PRIu64, std::numeric_limits<ui64>::max()));
    EXPECT_EQ("18446744073709551615", Format("%v", std::numeric_limits<ui64>::max()));
}

TEST(TFormatTest, HexIntegers)
{
    EXPECT_EQ("7b", Format("%x", 123));
    EXPECT_EQ("7B", Format("%X", 123));

    EXPECT_EQ("02a", Format("%03x", 42));
    EXPECT_EQ("2a", Format("%01x", 42));

    EXPECT_EQ("7fffffff", Format("%x", std::numeric_limits<i32>::max()));
    EXPECT_EQ("-80000000", Format("%x", std::numeric_limits<i32>::min()));

    EXPECT_EQ("0", Format("%x", 0U));
    EXPECT_EQ("0", Format("%X", 0U));
    EXPECT_EQ("ffffffff", Format("%x", std::numeric_limits<ui32>::max()));

    EXPECT_EQ("7fffffffffffffff", Format("%x", std::numeric_limits<i64>::max()));
    EXPECT_EQ("-8000000000000000", Format("%x", std::numeric_limits<i64>::min()));

    EXPECT_EQ("0", Format("%x", 0ULL));
    EXPECT_EQ("ffffffffffffffff", Format("%x", std::numeric_limits<ui64>::max()));
}

TEST(TFormatTest, Floats)
{
    EXPECT_EQ("3.14", Format("%.2f", 3.1415F));
    EXPECT_EQ("3.14", Format("%.2v", 3.1415F));
    EXPECT_EQ("3.14", Format("%.2lf", 3.1415));
    EXPECT_EQ("3.14", Format("%.2v", 3.1415));
    EXPECT_EQ(TString(std::to_string(std::numeric_limits<double>::max())),
            Format("%lF", std::numeric_limits<double>::max()));
}

TEST(TFormatTest, Bool)
{
    EXPECT_EQ("True", Format("%v", true));
    EXPECT_EQ("False", Format("%v", false));
    EXPECT_EQ("true", Format("%lv", true));
    EXPECT_EQ("false", Format("%lv", false));
}

TEST(TFormatTest, Quotes)
{
    EXPECT_EQ("\"True\"", Format("%Qv", true));
    EXPECT_EQ("'False'", Format("%qv", false));
    EXPECT_EQ("'\\\'\"'", Format("%qv", "\'\""));
    EXPECT_EQ("\"\\x01\"", Format("%Qv", "\x1"));
    EXPECT_EQ("'\\x1b'", Format("%qv", '\x1b'));
    EXPECT_EQ("'\\\\'", Format("%qv", '\\'));
    EXPECT_EQ("'\\n'", Format("%qv", '\n'));
    EXPECT_EQ("'\\t'", Format("%qv", '\t'));
    EXPECT_EQ("'\\\''", Format("%qv", '\''));
    EXPECT_EQ("\"'\"", Format("%Qv", '\''));
    EXPECT_EQ("'\"'", Format("%qv", '\"'));
    EXPECT_EQ("\"\\\"\"", Format("%Qv", '\"'));
}

TEST(TFormatTest, Escape)
{
    EXPECT_EQ("\'\"", Format("%hv", "\'\""));
    EXPECT_EQ("\\x01", Format("%hv", "\x1"));
    EXPECT_EQ("\\x1b", Format("%hv", '\x1b'));
    EXPECT_EQ("\\\\", Format("%hv", '\\'));
    EXPECT_EQ("\\n", Format("%hv", '\n'));
    EXPECT_EQ("\\t", Format("%hv", '\t'));
    EXPECT_EQ("\'", Format("%hv", '\''));
}

TEST(TFormatTest, Nullable)
{
    EXPECT_EQ("1", Format("%v", std::make_optional<int>(1)));
    EXPECT_EQ("<null>", Format("%v", std::nullopt));
    EXPECT_EQ("<null>", Format("%v", std::optional<int>()));
    EXPECT_EQ("3.14", Format("%.2f", std::optional<double>(3.1415)));
}

TEST(TFormatTest, Pointers)
{
    {
        auto ptr = reinterpret_cast<void*>(0x12345678);
        EXPECT_EQ("0x12345678", Format("%p", ptr));
        EXPECT_EQ("0x12345678", Format("%v", ptr));
        EXPECT_EQ("12345678", Format("%x", ptr));
        EXPECT_EQ("12345678", Format("%X", ptr));
    }
    {
        auto ptr = reinterpret_cast<void*>(0x12345678abcdefab);
        EXPECT_EQ("0x12345678abcdefab", Format("%p", ptr));
        EXPECT_EQ("0x12345678abcdefab", Format("%v", ptr));
        EXPECT_EQ("12345678abcdefab", Format("%x", ptr));
        EXPECT_EQ("12345678ABCDEFAB", Format("%X", ptr));
    }
}

TEST(TFormatTest, LazyMultiValueFormatter)
{
    int i = 1;
    TString s = "hello";
    std::vector<int> range{1, 2, 3};
    auto lazyFormatter = MakeLazyMultiValueFormatter(
        "int: %v, string: %v, range: %v",
        i,
        s,
        MakeFormattableView(range, TDefaultFormatter{}));
    EXPECT_EQ("int: 1, string: hello, range: [1, 2, 3]", Format("%v", lazyFormatter));
}

TEST(TFormatTest, VectorArg)
{
    std::vector<TString> params = {"a", "b", "c"};

    EXPECT_EQ(FormatVector("a is %v, b is %v, c is %v", params), "a is a, b is b, c is c");
}

TEST(TFormatTest, RuntimeFormat)
{
    TString format = "Hello %v";
    EXPECT_EQ(Format(TRuntimeFormat(format), "World"), "Hello World");
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
