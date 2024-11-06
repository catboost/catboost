#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/misc/compare.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TCompareTest, TernaryCompare)
{
    EXPECT_EQ(TernaryCompare(123, 123), 0);
    EXPECT_EQ(TernaryCompare(10, 20), -1);
    EXPECT_EQ(TernaryCompare(20, 10), +1);
    EXPECT_EQ(TernaryCompare(std::nan("1"), std::nan("1")), +1);
    EXPECT_EQ(TernaryCompare(std::nan("1"), std::nan("2")), +1);
    EXPECT_EQ(TernaryCompare(std::nan("1"), 123.0), +1);
    EXPECT_EQ(TernaryCompare(123.0, std::nan("1")), +1);
}

TEST(TCompareTest, NaNSafeTernaryCompare)
{
    EXPECT_EQ(NaNSafeTernaryCompare(std::nan("1"), std::nan("1")), 0);
    EXPECT_EQ(NaNSafeTernaryCompare(std::nan("1"), std::nan("2")), 0);
    EXPECT_EQ(NaNSafeTernaryCompare(123.0, std::nan("1")), -1);
    EXPECT_EQ(NaNSafeTernaryCompare(std::nan("1"), 123.0), +1);
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
class TTernaryCompareStringTest
    : public ::testing::Test
{ };

TYPED_TEST_SUITE_P(TTernaryCompareStringTest);

TYPED_TEST_P(TTernaryCompareStringTest, Compare)
{
    EXPECT_EQ(TernaryCompare(TypeParam("abc"), TypeParam("abc")), 0);
    EXPECT_EQ(TernaryCompare(TypeParam("x"), TypeParam("y")), -1);
    EXPECT_EQ(TernaryCompare(TypeParam("y"), TypeParam("x")), +1);
}

REGISTER_TYPED_TEST_SUITE_P(TTernaryCompareStringTest, Compare);

using TTernaryCompareStringTestTypes = ::testing::Types<
    TString,
    TStringBuf,
    std::string,
    std::string_view
>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    TypeParametrized,
    TTernaryCompareStringTest,
    TTernaryCompareStringTestTypes);

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
