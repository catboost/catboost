#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/coding/zig_zag.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TZigZagTest, Encode32)
{
    EXPECT_EQ(0u, ZigZagEncode32( 0));
    EXPECT_EQ(1u, ZigZagEncode32(-1));
    EXPECT_EQ(2u, ZigZagEncode32( 1));
    EXPECT_EQ(3u, ZigZagEncode32(-2));
    // ...
    EXPECT_EQ(std::numeric_limits<ui32>::max() - 1, ZigZagEncode32(std::numeric_limits<i32>::max()));
    EXPECT_EQ(std::numeric_limits<ui32>::max(),     ZigZagEncode32(std::numeric_limits<i32>::min()));
}

TEST(TZigZagTest, Decode32)
{
    EXPECT_EQ( 0, ZigZagDecode32(0));
    EXPECT_EQ(-1, ZigZagDecode32(1));
    EXPECT_EQ( 1, ZigZagDecode32(2));
    EXPECT_EQ(-2, ZigZagDecode32(3));
    // ...
    EXPECT_EQ(std::numeric_limits<i32>::max(), ZigZagDecode32(std::numeric_limits<ui32>::max() - 1));
    EXPECT_EQ(std::numeric_limits<i32>::min(), ZigZagDecode32(std::numeric_limits<ui32>::max()));
}

TEST(TZigZagTest, Encode64)
{
    EXPECT_EQ(0ull, ZigZagEncode64( 0));
    EXPECT_EQ(1ull, ZigZagEncode64(-1));
    EXPECT_EQ(2ull, ZigZagEncode64( 1));
    EXPECT_EQ(3ull, ZigZagEncode64(-2));
    // ...
    EXPECT_EQ(std::numeric_limits<ui64>::max() - 1, ZigZagEncode64(std::numeric_limits<i64>::max()));
    EXPECT_EQ(std::numeric_limits<ui64>::max(),     ZigZagEncode64(std::numeric_limits<i64>::min()));
}

TEST(TZigZagTest, Decode64)
{
    EXPECT_EQ(ZigZagDecode64(0),  0ll);
    EXPECT_EQ(ZigZagDecode64(1), -1ll);
    EXPECT_EQ(ZigZagDecode64(2),  1ll);
    EXPECT_EQ(ZigZagDecode64(3), -2ll);
    // ...
    EXPECT_EQ(std::numeric_limits<i64>::max(), ZigZagDecode64(std::numeric_limits<ui64>::max() - 1));
    EXPECT_EQ(std::numeric_limits<i64>::min(), ZigZagDecode64(std::numeric_limits<ui64>::max()));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
