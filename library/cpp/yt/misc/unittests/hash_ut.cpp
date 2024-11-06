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

} // namespace
} // namespace NYT
