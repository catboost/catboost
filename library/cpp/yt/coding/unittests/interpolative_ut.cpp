#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/coding/interpolative.h>
#include <library/cpp/yt/coding/varint.h>

#include <algorithm>
#include <bit>
#include <numeric>
#include <random>
#include <set>
#include <vector>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////
// Truncated binary

TEST(TTruncatedBinaryTest, Exhaustive)
{
    for (ui32 rangeSize = 1; rangeSize <= 2050; ++rangeSize) {
        for (ui32 value = 0; value < rangeSize; ++value) {
            std::vector<char> buffer(16, 0);
            TBitWriter writer(buffer.data());
            NInterpolativeCodingDetail::WriteTruncatedBinary(&writer, value, rangeSize);
            writer.Finish();

            TBitReader reader(buffer.data());
            EXPECT_EQ(NInterpolativeCodingDetail::ReadTruncatedBinary(&reader, rangeSize), value)
                << "rangeSize=" << rangeSize << " value=" << value;
        }
    }
}

TEST(TTruncatedBinaryTest, MinimalLength)
{
    // Every codeword is floor(log2(rangeSize)) or ceil(log2(rangeSize)) bits.
    for (ui32 rangeSize = 1; rangeSize <= 1000; ++rangeSize) {
        int lowWidth = std::bit_width(rangeSize) - 1;
        for (ui32 value = 0; value < rangeSize; ++value) {
            std::vector<char> buffer(16, 0);
            TBitWriter writer(buffer.data());
            NInterpolativeCodingDetail::WriteTruncatedBinary(&writer, value, rangeSize);
            char* end = writer.Finish();

            TBitReader reader(buffer.data());
            NInterpolativeCodingDetail::ReadTruncatedBinary(&reader, rangeSize);
            const char* readEnd = reader.Finish();
            i64 bytes = end - buffer.data();
            // Read must consume no more bytes than were written.
            EXPECT_LE(readEnd - buffer.data(), bytes);
            EXPECT_LE(bytes, (lowWidth + 1 + 7) / 8 + 1);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Interpolative coding

template <class T>
std::vector<char> Encode(const std::vector<T>& values, ui32 lo, ui32 hi)
{
    std::vector<char> buffer(values.size() * sizeof(ui32) + 16, 0);
    TBitWriter writer(buffer.data());
    InterpolativeEncode(&writer, TRange(values), lo, hi);
    char* end = writer.Finish();
    buffer.resize(end - buffer.data());
    return buffer;
}

template <class T>
std::vector<T> Decode(std::vector<char> buffer, int count, ui32 lo, ui32 hi)
{
    buffer.resize(buffer.size() + 8, 0);  // reader may over-read up to 8 bytes
    std::vector<T> values(count);
    TBitReader reader(buffer.data());
    InterpolativeDecode(&reader, TMutableRange(values), lo, hi);
    return values;
}

template <class T>
void ExpectRoundTrip(const std::vector<T>& values, ui32 lo, ui32 hi)
{
    auto decoded = Decode<T>(Encode(values, lo, hi), std::ssize(values), lo, hi);
    EXPECT_EQ(decoded, values);
}

TEST(TInterpolativeCodingTest, Empty)
{
    ExpectRoundTrip<ui32>({}, 0, 100);
}

TEST(TInterpolativeCodingTest, Single)
{
    ExpectRoundTrip<ui32>({0}, 0, 0);
    ExpectRoundTrip<ui32>({42}, 0, 100);
    ExpectRoundTrip<ui32>({100}, 0, 100);
}

TEST(TInterpolativeCodingTest, SmallCases)
{
    ExpectRoundTrip<ui32>({3, 7}, 0, 10);
    ExpectRoundTrip<ui32>({0, 1, 2}, 0, 2);          // full, zero bits
    ExpectRoundTrip<ui32>({0, 5, 10}, 0, 10);        // boundaries present
    ExpectRoundTrip<ui32>({1, 2, 3, 4, 5}, 0, 6);
}

TEST(TInterpolativeCodingTest, FullRange)
{
    // Every value present => every range collapses to a singleton (zero bits).
    std::vector<ui32> values(500);
    std::iota(values.begin(), values.end(), 7);
    auto encoded = Encode(values, 7, 506);
    EXPECT_TRUE(encoded.empty());
    EXPECT_EQ(Decode<ui32>(encoded, 500, 7, 506), values);
}

TEST(TInterpolativeCodingTest, RandomRoundTrip)
{
    std::mt19937 rng(12345);
    for (int iteration = 0; iteration < 500; ++iteration) {
        ui32 universe = 1 + rng() % 200'000;
        int count = std::min<ui32>(universe, 1 + rng() % 2000);
        std::set<ui32> unique;
        std::uniform_int_distribution<ui32> dist(0, universe - 1);
        while (std::ssize(unique) < count) {
            unique.insert(dist(rng));
        }
        std::vector<ui32> values(unique.begin(), unique.end());
        ExpectRoundTrip<ui32>(values, 0, universe - 1);
    }
}

TEST(TInterpolativeCodingTest, NonZeroLowerBound)
{
    std::mt19937 rng(999);
    for (int iteration = 0; iteration < 200; ++iteration) {
        ui32 lo = rng() % 100'000;
        ui32 span = 1 + rng() % 100'000;
        ui32 hi = lo + span;
        int count = std::min<ui32>(span + 1, 1 + rng() % 500);
        std::set<ui32> unique;
        std::uniform_int_distribution<ui32> dist(lo, hi);
        while (std::ssize(unique) < count) {
            unique.insert(dist(rng));
        }
        std::vector<ui32> values(unique.begin(), unique.end());
        ExpectRoundTrip<ui32>(values, lo, hi);
    }
}

TEST(TInterpolativeCodingTest, Ui64Values)
{
    std::vector<ui64> values = {0, 1, 100, 1000, 50'000, 200'000};
    ExpectRoundTrip<ui64>(values, 0, 200'000);
}

TEST(TInterpolativeCodingTest, MaxByteSize)
{
    std::mt19937 rng(555);
    for (int iteration = 0; iteration < 300; ++iteration) {
        ui32 universe = 1 + rng() % 200'000;
        int count = std::min<ui32>(universe, 1 + rng() % 1000);
        std::set<ui32> unique;
        std::uniform_int_distribution<ui32> dist(0, universe - 1);
        while (std::ssize(unique) < count) {
            unique.insert(dist(rng));
        }
        std::vector<ui32> values(unique.begin(), unique.end());

        size_t bound = GetInterpolativeMaxByteSize(count, 0, universe - 1);
        std::vector<char> buffer(bound, 0);
        TBitWriter writer(buffer.data());
        InterpolativeEncode(&writer, TRange(values), 0, universe - 1);
        EXPECT_LE(static_cast<size_t>(writer.Finish() - buffer.data()), bound);
    }
}

TEST(TInterpolativeCodingTest, MultipleListsInOneBuffer)
{
    // Mirrors real usage: each list is prefixed with a varint length and is
    // byte-aligned, so lists can be concatenated and read back in sequence.
    std::mt19937 rng(7);
    ui32 hi = 199999;
    std::vector<std::vector<ui32>> lists;
    for (int listIndex = 0; listIndex < 50; ++listIndex) {
        int count = 1 + rng() % 300;
        std::set<ui32> unique;
        std::uniform_int_distribution<ui32> dist(0, hi);
        while (std::ssize(unique) < count) {
            unique.insert(dist(rng));
        }
        lists.emplace_back(unique.begin(), unique.end());
    }

    std::vector<char> buffer(200'000, 0);
    char* ptr = buffer.data();
    for (const auto& list : lists) {
        ptr += WriteVarUint32(ptr, static_cast<ui32>(list.size()));
        TBitWriter writer(ptr);
        InterpolativeEncode(&writer, TRange(list), 0, hi);
        ptr = writer.Finish();
    }

    const char* readPtr = buffer.data();
    for (const auto& list : lists) {
        ui32 count;
        readPtr += ReadVarUint32(readPtr, &count);
        ASSERT_EQ(count, list.size());
        std::vector<ui32> decoded(count);
        TBitReader reader(readPtr);
        InterpolativeDecode(&reader, TMutableRange(decoded), 0, hi);
        readPtr = reader.Finish();
        EXPECT_EQ(decoded, list);
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
