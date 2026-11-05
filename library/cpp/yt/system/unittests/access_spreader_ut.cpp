#include <library/cpp/yt/system/access_spreader.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <array>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TAccessSpreaderTest, ReturnsStripeWithinRequestedCount)
{
    EXPECT_EQ(GetCurrentAccessStripe(1), 0);

    constexpr std::array<int, 5> StripeCounts{2, 3, 7, 64, 100};
    for (int stripeCount : StripeCounts) {
        for (int iteration = 0; iteration < 64; ++iteration) {
            int stripeIndex = GetCurrentAccessStripe(stripeCount);
            EXPECT_GE(stripeIndex, 0);
            EXPECT_LT(stripeIndex, stripeCount);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
