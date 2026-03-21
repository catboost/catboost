#include <library/cpp/yt/misc/range_helpers.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <list>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TRangeHelpersTest, ZipMutable)
{
    std::vector<int> vectorA(4);
    std::vector<int> vectorB = {1, 2, 3};
    for (auto [a, b] : ZipMutable(vectorA, vectorB)) {
        *a = *b + 1;
    }

    auto expectedA = std::vector<int>{2, 3, 4, 0};
    EXPECT_EQ(expectedA, vectorA);
}

////////////////////////////////////////////////////////////////////////////////

TEST(TRangeHelpersTest, RangeToVector)
{
    auto data = std::vector<std::string>{"A", "B", "C", "D"};
    auto range = std::ranges::views::transform(data, [] (std::string x) {
        return "_" + x;
    });

    std::initializer_list<std::string> expectedValues{"_A", "_B", "_C", "_D"};
    EXPECT_EQ(std::vector<std::string>(expectedValues), RangeTo<std::vector<std::string>>(range));
    using TListStrings = std::list<std::string>;
    EXPECT_EQ(TListStrings(expectedValues), RangeTo<TListStrings>(range));
}

TEST(TRangeHelpersTest, RangeToString)
{
    auto data = "_sample_"sv;
    auto range = std::ranges::views::filter(data, [] (char x) {
        return x != '_';
    });
    auto expectedData = "sample"sv;

    EXPECT_EQ(std::string(expectedData), RangeTo<std::string>(range));
    EXPECT_EQ(TString(expectedData), RangeTo<TString>(range));
}

////////////////////////////////////////////////////////////////////////////////

TEST(TRangeHelpersTest, MonadicRangeToVector)
{
    auto data = std::vector<std::string>{"A", "B", "C", "D"};
    auto range = std::ranges::views::transform(data, [] (std::string x) {
        return "_" + x;
    });

    std::initializer_list<std::string> expectedValues{"_A", "_B", "_C", "_D"};
    EXPECT_EQ(std::vector<std::string>(expectedValues), range | RangeTo<std::vector<std::string>>());
    using TListStrings = std::list<std::string>;
    EXPECT_EQ(TListStrings(expectedValues), range  | RangeTo<TListStrings>());
}

TEST(TRangeHelpersTest, MonadicRangeToString)
{
    auto data = "_sample_"sv;
    auto range = std::ranges::views::filter(data, [] (char x) {
        return x != '_';
    });
    auto expectedData = "sample"sv;

    EXPECT_EQ(std::string(expectedData), range | RangeTo<std::string>());
    EXPECT_EQ(TString(expectedData), range | RangeTo<TString>());
}

////////////////////////////////////////////////////////////////////////////////

TEST(TRangeHelpersTest, Fold)
{
    EXPECT_EQ(0, FoldRange(std::vector<int>{}, std::plus{}));
    EXPECT_EQ(6, FoldRange(std::vector<int>{1, 2, 3}, std::plus{}));
    EXPECT_EQ(5, FoldRange(
        std::vector<std::vector<int>>{{1, 2}, {3, 4, 5}},
        std::plus{},
        std::ranges::ssize));
}

////////////////////////////////////////////////////////////////////////////////

TEST(TRangeHelpersTest, StaticRangeToVector)
{
    EXPECT_EQ(StaticRangeTo<std::vector<int>>(1), std::vector<int>{1});
    auto expected = std::vector<int>{1, 2, 10};
    auto result = StaticRangeTo<std::vector<int>>(1, 2, 10);
    EXPECT_EQ(result, expected);
}

TEST(TRangeHelpersTest, StaticRangeToVectorMoveOnly)
{
    auto result = StaticRangeTo<std::vector<std::unique_ptr<int>>>(std::make_unique<int>(1), std::make_unique<int>(2));
    ASSERT_EQ(std::ssize(result), 2);
    EXPECT_EQ(*result[0], 1);
    EXPECT_EQ(*result[1], 2);
}

TEST(TRangeHelpersTest, TStaticRangeToVector)
{
    EXPECT_EQ(static_cast<std::vector<int>>(TStaticRange{1}), std::vector<int>{1});
    auto expected = std::vector<int>{1, 2, 10};
    std::vector<int> result = TStaticRange{1, 2, 10};
    EXPECT_EQ(result, expected);
}

TEST(TRangeHelpersTest, TStaticRangeToVectorMoveOnly)
{
    std::vector<std::unique_ptr<int>> result = TStaticRange(std::make_unique<int>(1), std::make_unique<int>(2));
    ASSERT_EQ(std::ssize(result), 2);
    EXPECT_EQ(*result[0], 1);
    EXPECT_EQ(*result[1], 2);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
