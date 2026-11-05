#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/system/env.h>

#include <library/cpp/yt/misc/guid.h>

#include <library/cpp/yt/string/format.h>

#include <util/generic/scope.h>

#include <util/system/env.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TParseEnvironNameValuePairTest, NonNull)
{
    auto pair = ParseEnvironNameValuePair("var=value");
    EXPECT_EQ(pair.first, "var");
    EXPECT_EQ(pair.second, "value");
}

TEST(TParseEnvironNameValuePairTest, Null)
{
    auto pair = ParseEnvironNameValuePair("some");
    EXPECT_EQ(pair.first, "some");
    EXPECT_EQ(pair.second, std::nullopt);
}

////////////////////////////////////////////////////////////////////////////////

#if defined(_linux_) || defined(_darwin_)

TEST(TGetEnvironNameValuePairsTest, Simple)
{
    auto key = Format("var%v", TGuid::Create());
    auto value = TString("value");
    auto pair = Format("%v=%v", key, value);
    SetEnv(key, value);
    auto pairs = GetEnvironNameValuePairs();
    EXPECT_TRUE(std::ranges::find(pairs, pair) != pairs.end());
    UnsetEnv(key);
}

TEST(TGetEnvValueOrThrowTest, Typed)
{
    auto key = Format("var%v", TGuid::Create());
    SetEnv(key, "42");
    Y_DEFER {
        UnsetEnv(key);
    };
    EXPECT_EQ(GetEnvValueOrThrow<int>(key), 42);
    EXPECT_EQ(GetEnvValueOrThrow(key), "42");
}

TEST(TGetEnvValueOrThrowTest, Missing)
{
    auto key = Format("var%v", TGuid::Create());
    EXPECT_THROW(GetEnvValueOrThrow<int>(key), std::exception);
}

TEST(TGetEnvValueOrThrowTest, Unparseable)
{
    auto key = Format("var%v", TGuid::Create());
    SetEnv(key, "not-a-number");
    Y_DEFER {
        UnsetEnv(key);
    };
    EXPECT_THROW(GetEnvValueOrThrow<int>(key), std::exception);
}

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT


