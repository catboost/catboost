#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/misc/lazy.h>

#include <string>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TLazyTest, UnlazyPassesPlainValuesThrough)
{
    EXPECT_EQ(Unlazy(123), 123);

    int value = 42;
    EXPECT_EQ(&Unlazy(value), &value);
}

TEST(TLazyTest, UnlazyEvaluatesLazyValues)
{
    int calls = 0;
    auto lazy = YT_LAZY((++calls, 123));
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(Unlazy(lazy), 123);
    EXPECT_EQ(calls, 1);
}

TEST(TLazyTest, CapturesEnclosingScope)
{
    std::string value = "hello";
    const std::string* pointer = &value;
    EXPECT_EQ(Unlazy(YT_LAZY(pointer->size())), 5u);
}

TEST(TLazyTest, Concept)
{
    static_assert(!CLazy<int>);
    static_assert(CLazy<decltype(YT_LAZY(123))>);
}

TEST(TLazyTest, UnlazyType)
{
    // NB: Non-lazy types must come through verbatim, references included; see #TUnlazy.
    static_assert(std::same_as<TUnlazy<int>, int>);
    static_assert(std::same_as<TUnlazy<int&>, int&>);
    static_assert(std::same_as<TUnlazy<const std::string&>, const std::string&>);

    static_assert(std::same_as<TUnlazy<decltype(YT_LAZY(123))>, int>);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
