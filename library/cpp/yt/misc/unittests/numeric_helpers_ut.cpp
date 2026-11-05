#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/misc/numeric_helpers.h>

#include <util/generic/ymath.h>

#include <util/system/types.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TNumericHelpersTest, DivCeil)
{
    EXPECT_EQ(DivCeil(0, 3), 0);
    EXPECT_EQ(DivCeil(1, 3), 1);
    EXPECT_EQ(DivCeil(2, 3), 1);
    EXPECT_EQ(DivCeil(3, 3), 1);
    EXPECT_EQ(DivCeil(4, 3), 2);
    EXPECT_EQ(DivCeil(7, 3), 3);
    EXPECT_EQ(DivCeil(10, 1), 10);
}

TEST(TNumericHelpersTest, DivCeilNegative)
{
    EXPECT_EQ(DivCeil(-1, 3), 0);
    EXPECT_EQ(DivCeil(-2, 3), 0);
    EXPECT_EQ(DivCeil(-3, 3), -1);
    EXPECT_EQ(DivCeil(-4, 3), -1);
    EXPECT_EQ(DivCeil(-7, 3), -2);
}

TEST(TNumericHelpersTest, DivCeilWideTypes)
{
    EXPECT_EQ(DivCeil<i64>(1, 1ll << 40), 1);
    EXPECT_EQ(DivCeil<i64>((1ll << 40) + 1, 1ll << 40), 2);
    EXPECT_EQ(DivCeil<i64>(Max<i64>(), Max<i64>()), 1);
    EXPECT_EQ(DivCeil<long>(5, 2), 3);
}

TEST(TNumericHelpersTest, DivCeilZeroDenominator)
{
    EXPECT_DEATH(Y_UNUSED(DivCeil(1, 0)), ".*");
}

TEST(TNumericHelpersTest, DivRoundOddDenominator)
{
    EXPECT_EQ(DivRound(0, 5), 0);
    EXPECT_EQ(DivRound(1, 5), 0);
    EXPECT_EQ(DivRound(2, 5), 0);
    EXPECT_EQ(DivRound(3, 5), 1);
    EXPECT_EQ(DivRound(5, 5), 1);
    EXPECT_EQ(DivRound(7, 5), 1);
    EXPECT_EQ(DivRound(8, 5), 2);
}

TEST(TNumericHelpersTest, DivRoundEvenDenominator)
{
    EXPECT_EQ(DivRound(1, 4), 0);
    EXPECT_EQ(DivRound(2, 4), 1);
    EXPECT_EQ(DivRound(3, 4), 1);
    EXPECT_EQ(DivRound(5, 4), 1);
    EXPECT_EQ(DivRound(6, 4), 2);
}

TEST(TNumericHelpersTest, DivRoundNegative)
{
    // NB: The remainder is truncated towards zero, hence the rounding is not symmetric.
    EXPECT_EQ(DivRound(-2, 5), 0);
    EXPECT_EQ(DivRound(-3, 5), 0);
    EXPECT_EQ(DivRound(-7, 5), -1);
    EXPECT_EQ(DivRound(-8, 5), -1);
    EXPECT_EQ(DivRound(-10, 5), -2);
}

TEST(TNumericHelpersTest, RoundUp)
{
    EXPECT_EQ(RoundUp(0, 8), 0);
    EXPECT_EQ(RoundUp(1, 8), 8);
    EXPECT_EQ(RoundUp(7, 8), 8);
    EXPECT_EQ(RoundUp(8, 8), 8);
    EXPECT_EQ(RoundUp(9, 8), 16);
    EXPECT_EQ(RoundUp<i64>(1, 1ll << 40), 1ll << 40);
}

TEST(TNumericHelpersTest, RoundDown)
{
    EXPECT_EQ(RoundDown(0, 8), 0);
    EXPECT_EQ(RoundDown(1, 8), 0);
    EXPECT_EQ(RoundDown(7, 8), 0);
    EXPECT_EQ(RoundDown(8, 8), 8);
    EXPECT_EQ(RoundDown(9, 8), 8);
    EXPECT_EQ(RoundDown<i64>((1ll << 40) + 1, 1ll << 40), 1ll << 40);
}

TEST(TNumericHelpersTest, RoundNegative)
{
    // NB: Both helpers truncate towards zero rather than towards minus infinity.
    EXPECT_EQ(RoundUp(-9, 8), -8);
    EXPECT_EQ(RoundDown(-9, 8), -8);
    EXPECT_EQ(RoundUp(-8, 8), -8);
    EXPECT_EQ(RoundDown(-8, 8), -8);
}

TEST(TNumericHelpersTest, GetSign)
{
    EXPECT_EQ(GetSign(0), 0);
    EXPECT_EQ(GetSign(1), 1);
    EXPECT_EQ(GetSign(-1), -1);
    EXPECT_EQ(GetSign(Max<i64>()), 1);
    EXPECT_EQ(GetSign(Min<i64>()), -1);

    EXPECT_EQ(GetSign(0.0), 0);
    EXPECT_EQ(GetSign(-0.0), 0);
    EXPECT_EQ(GetSign(-0.5), -1);
    EXPECT_EQ(GetSign(0.5), 1);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
