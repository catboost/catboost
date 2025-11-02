#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/system/env.h>

#include <library/cpp/yt/misc/guid.h>

#include <library/cpp/yt/string/format.h>

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

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT


