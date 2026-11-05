#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/coding/bit_io.h>

#include <utility>
#include <vector>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TBitIOTest, RoundTrip)
{
    std::vector<std::pair<ui32, int>> items = {
        {0, 0}, {1, 1}, {0, 1}, {5, 3}, {0, 8}, {255, 8}, {12345, 14},
        {0, 32}, {0xffffffffu, 32}, {0x12345, 17}, {7, 3}, {1u << 30, 31},
    };
    std::vector<char> buffer(256, 0);

    TBitWriter writer(buffer.data());
    for (auto [value, width] : items) {
        writer.WriteBits(value, width);
    }
    writer.Finish();

    TBitReader reader(buffer.data());
    for (auto [value, width] : items) {
        EXPECT_EQ(reader.ReadBits(width), value) << "width=" << width;
    }
}

TEST(TBitIOTest, ZeroWidthIsNoop)
{
    std::vector<char> buffer(16, 0);
    TBitWriter writer(buffer.data());
    writer.WriteBits(0, 0);
    writer.WriteBits(1, 1);
    writer.WriteBits(0, 0);
    char* end = writer.Finish();
    EXPECT_EQ(end - buffer.data(), 1);  // a single bit occupies one byte

    TBitReader reader(buffer.data());
    EXPECT_EQ(reader.ReadBits(0), 0u);
    EXPECT_EQ(reader.ReadBits(1), 1u);
    EXPECT_EQ(reader.ReadBits(0), 0u);
}

TEST(TBitIOTest, FlushBoundary)
{
    // Many 17-bit writes repeatedly cross the internal 32-bit flush boundary.
    constexpr int Count = 1000;
    std::vector<char> buffer(Count * 3 + 16, 0);
    TBitWriter writer(buffer.data());
    for (int i = 0; i < Count; ++i) {
        writer.WriteBits(static_cast<ui32>(i) & 0x1ffff, 17);
    }
    writer.Finish();

    TBitReader reader(buffer.data());
    for (int i = 0; i < Count; ++i) {
        EXPECT_EQ(reader.ReadBits(17), static_cast<ui32>(i) & 0x1ffff) << "i=" << i;
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
