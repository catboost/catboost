#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/coding/varint.h>

#include <util/random/random.h>

#include <util/stream/mem.h>

#include <util/string/escape.h>

#include <tuple>

namespace NYT {
namespace {

using ::testing::Values;

////////////////////////////////////////////////////////////////////////////////

class TWriteVarIntTest
    : public ::testing::TestWithParam<std::tuple<ui64, std::string>>
{ };

TEST_P(TWriteVarIntTest, Serialization)
{
    ui64 value = std::get<0>(GetParam());
    std::string rightAnswer = std::get<1>(GetParam());

    TStringStream outputStream;
    WriteVarUint64(&outputStream, value);
    EXPECT_EQ(rightAnswer, outputStream.Str());
}

////////////////////////////////////////////////////////////////////////////////

class TReadVarIntTest: public ::testing::TestWithParam<std::tuple<ui64, std::string> >
{ };

TEST_P(TReadVarIntTest, Serialization)
{
    ui64 rightAnswer = std::get<0>(GetParam());
    const auto input = TString(std::get<1>(GetParam()));

    TStringInput inputStream(input);
    ui64 value;
    EXPECT_EQ(std::ssize(input), ReadVarUint64(&inputStream, &value));
    EXPECT_EQ(rightAnswer, value);

    EXPECT_EQ(std::ssize(input), ReadVarUint64(input.data(), &value));
    EXPECT_EQ(rightAnswer, value);

    EXPECT_EQ(std::ssize(input), ReadVarUint64(input.data(), input.data() + input.size(), &value));
    EXPECT_EQ(rightAnswer, value);

    int position = 0;
    EXPECT_EQ(std::ssize(input), ReadVarUint64([&] { return input[position++]; }, &value));
    EXPECT_EQ(rightAnswer, value);
}

TEST(TReadVarIntTest, Overflow)
{
    const TString input("\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x01", 11);
    TStringInput inputStream(input);
    ui64 value;
    EXPECT_ANY_THROW(ReadVarUint64(&inputStream, &value));

    int position = 0;
    EXPECT_ANY_THROW(ReadVarUint64([&] { return input[position++]; }, &value));
}

////////////////////////////////////////////////////////////////////////////////

auto ValuesForVarIntTests = Values(
    // Simple cases.
    std::make_tuple(0x0ull,                std::string("\x00", 1)),
    std::make_tuple(0x1ull,                std::string("\x01", 1)),
    std::make_tuple(0x2ull,                std::string("\x02", 1)),
    std::make_tuple(0x3ull,                std::string("\x03", 1)),
    std::make_tuple(0x4ull,                std::string("\x04", 1)),

    // The following "magic numbers" are critical points for varint encoding.
    std::make_tuple((1ull << 7) - 1,       std::string("\x7f", 1)),
    std::make_tuple((1ull << 7),           std::string("\x80\x01", 2)),
    std::make_tuple((1ull << 14) - 1,      std::string("\xff\x7f", 2)),
    std::make_tuple((1ull << 14),          std::string("\x80\x80\x01", 3)),
    std::make_tuple((1ull << 21) - 1,      std::string("\xff\xff\x7f", 3)),
    std::make_tuple((1ull << 21),          std::string("\x80\x80\x80\x01", 4)),
    std::make_tuple((1ull << 28) - 1,      std::string("\xff\xff\xff\x7f", 4)),
    std::make_tuple((1ull << 28),          std::string("\x80\x80\x80\x80\x01", 5)),
    std::make_tuple((1ull << 35) - 1,      std::string("\xff\xff\xff\xff\x7f", 5)),
    std::make_tuple((1ull << 35),          std::string("\x80\x80\x80\x80\x80\x01", 6)),
    std::make_tuple((1ull << 42) - 1,      std::string("\xff\xff\xff\xff\xff\x7f", 6)),
    std::make_tuple((1ull << 42),          std::string("\x80\x80\x80\x80\x80\x80\x01", 7)),
    std::make_tuple((1ull << 49) - 1,      std::string("\xff\xff\xff\xff\xff\xff\x7f", 7)),
    std::make_tuple((1ull << 49),          std::string("\x80\x80\x80\x80\x80\x80\x80\x01", 8)),
    std::make_tuple((1ull << 56) - 1,      std::string("\xff\xff\xff\xff\xff\xff\xff\x7f", 8)),
    std::make_tuple((1ull << 56),          std::string("\x80\x80\x80\x80\x80\x80\x80\x80\x01", 9)),
    std::make_tuple((1ull << 63) - 1,      std::string("\xff\xff\xff\xff\xff\xff\xff\xff\x7f", 9)),
    std::make_tuple((1ull << 63),          std::string("\x80\x80\x80\x80\x80\x80\x80\x80\x80\x01", 10)),

    // Boundary case.
    std::make_tuple(static_cast<ui64>(-1), std::string("\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01", 10))
);

INSTANTIATE_TEST_SUITE_P(ValueParametrized, TWriteVarIntTest,
    ValuesForVarIntTests);

INSTANTIATE_TEST_SUITE_P(ValueParametrized, TReadVarIntTest,
    ValuesForVarIntTests);

////////////////////////////////////////////////////////////////////////////////

TEST(TVarInt32Test, RandomValues)
{
    srand(100500); // Set seed
    const int numberOfValues = 10000;

    TStringStream stream;
    int position = 0;
    for (int i = 0; i < numberOfValues; ++i) {
        i32 expected = static_cast<i32>(RandomNumber<ui32>());
        WriteVarInt32(&stream, expected);
        const auto& bytes = stream.Str();
        i32 actual;

        int bytesRead = ReadVarInt32(&stream, &actual);
        EXPECT_EQ(expected, actual)
            << "Encoded Variant: " << EscapeC(bytes);

        EXPECT_EQ(bytesRead, ReadVarInt32(bytes.data() + position, &actual));
        EXPECT_EQ(expected, actual)
            << "Encoded Variant: " << EscapeC(bytes);

        EXPECT_EQ(bytesRead, ReadVarInt32(bytes.data() + position, bytes.data() + bytes.size(), &actual));
        EXPECT_EQ(expected, actual)
            << "Encoded Variant: " << EscapeC(bytes);

        int callbackPosition = position;
        EXPECT_EQ(bytesRead, ReadVarInt32([&] { return bytes[callbackPosition++]; }, &actual));
        EXPECT_EQ(expected, actual)
            << "Encoded Variant: " << EscapeC(bytes);

        position += bytesRead;
    }
}

////////////////////////////////////////////////////////////////////////////////

TEST(TVarUint32Test, BoundaryValues)
{
    ui32 value = 0;

    // UINT32_MAX encoded as a 5-byte varint: should succeed.
    {
        const char maxBytes[] = "\xff\xff\xff\xff\x0f";
        TMemoryInput maxInput(maxBytes, 5);
        EXPECT_EQ(5, ReadVarUint32(&maxInput, &value));
        EXPECT_EQ(std::numeric_limits<ui32>::max(), value);
        EXPECT_EQ(5, ReadVarUint32(maxBytes, &value));
        EXPECT_EQ(std::numeric_limits<ui32>::max(), value);
        EXPECT_EQ(5, ReadVarUint32(maxBytes, maxBytes + 5, &value));
        EXPECT_EQ(std::numeric_limits<ui32>::max(), value);
        int position = 0;
        EXPECT_EQ(5, ReadVarUint32([&] { return maxBytes[position++]; }, &value));
        EXPECT_EQ(std::numeric_limits<ui32>::max(), value);
    }

    // 2^32 encoded as a 5-byte varint: should throw.
    {
        const char overflowBytes[] = "\x80\x80\x80\x80\x10";
        TMemoryInput overflowInput(overflowBytes, 5);
        EXPECT_THROW(ReadVarUint32(&overflowInput, &value), TSimpleException);
        EXPECT_THROW(ReadVarUint32(overflowBytes, &value), TSimpleException);
        EXPECT_THROW(ReadVarUint32(overflowBytes, overflowBytes + 5, &value), TSimpleException);
        int position = 0;
        EXPECT_THROW(ReadVarUint32([&] { return overflowBytes[position++]; }, &value), TSimpleException);
    }

    // Max 5-byte varint (2^35 - 1): should throw.
    {
        const char maxFiveByteBytes[] = "\xff\xff\xff\xff\x7f";
        TMemoryInput maxFiveByteInput(maxFiveByteBytes, 5);
        EXPECT_THROW(ReadVarUint32(&maxFiveByteInput, &value), TSimpleException);
        EXPECT_THROW(ReadVarUint32(maxFiveByteBytes, &value), TSimpleException);
        EXPECT_THROW(ReadVarUint32(maxFiveByteBytes, maxFiveByteBytes + 5, &value), TSimpleException);
        int position = 0;
        EXPECT_THROW(ReadVarUint32([&] { return maxFiveByteBytes[position++]; }, &value), TSimpleException);
    }
}

////////////////////////////////////////////////////////////////////////////////

TEST(TVarUint64Test, BoundaryValues)
{
    // UINT64_MAX encoded as a 10-byte varint: should succeed.
    const char maxBytes[] = "\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01";
    ui64 value = 0;
    EXPECT_EQ(10, ReadVarUint64(maxBytes, &value));
    EXPECT_EQ(std::numeric_limits<ui64>::max(), value);

    // 2^64 encoded as a 10-byte varint: should throw.
    const char overflowBytes[] = "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x02";
    EXPECT_THROW(ReadVarUint64(overflowBytes, &value), TSimpleException);

    // Max 10-byte varint (2^70 - 1): should throw.
    const char maxTenByteBytes[] = "\xff\xff\xff\xff\xff\xff\xff\xff\xff\x7f";
    EXPECT_THROW(ReadVarUint64(maxTenByteBytes, &value), TSimpleException);
}

////////////////////////////////////////////////////////////////////////////////

TEST(TVarInt64Test, RandomValues)
{
    srand(100500); // Set seed
    const int numberOfValues = 10000;

    TStringStream stream;
    int position = 0;
    for (int i = 0; i < numberOfValues; ++i) {
        i64 expected = static_cast<i64>(RandomNumber<ui64>());
        WriteVarInt64(&stream, expected);
        const auto& bytes = stream.Str();
        i64 actual;

        int bytesRead = ReadVarInt64(&stream, &actual);
        EXPECT_EQ(expected, actual)
            << "Encoded Variant: " << EscapeC(bytes);

        EXPECT_EQ(bytesRead, ReadVarInt64(bytes.data() + position, &actual));
        EXPECT_EQ(expected, actual)
            << "Encoded Variant: " << EscapeC(bytes);

        EXPECT_EQ(bytesRead, ReadVarInt64(bytes.data() + position, bytes.data() + bytes.size(), &actual));
        EXPECT_EQ(expected, actual)
            << "Encoded Variant: " << EscapeC(bytes);

        int callbackPosition = position;
        EXPECT_EQ(bytesRead, ReadVarInt64([&] { return bytes[callbackPosition++]; }, &actual));
        EXPECT_EQ(expected, actual)
            << "Encoded Variant: " << EscapeC(bytes);

        position += bytesRead;
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
