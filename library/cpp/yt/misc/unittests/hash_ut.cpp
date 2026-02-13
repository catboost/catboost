#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/misc/hash.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(THashTest, NaNSafeHash)
{
    EXPECT_EQ(NaNSafeHash(123), THash<int>()(123));
    EXPECT_EQ(NaNSafeHash(std::nan("1")), NaNSafeHash(std::nan("2")));
}

////////////////////////////////////////////////////////////////////////////////

TEST(THashTest, SplitMix64Test)
{
    EXPECT_EQ(SplitMix64(0), 0xe220a8397b1dcdafULL);
    EXPECT_EQ(SplitMix64(12345), 0x22118258a9d111a0ULL);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
