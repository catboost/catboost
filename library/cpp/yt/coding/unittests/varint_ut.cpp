#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/coding/varint.h>

#include <util/random/random.h>

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
    auto input = TString(std::get<1>(GetParam()));

    TStringInput inputStream(input);
    ui64 value;
    ReadVarUint64(&inputStream, &value);
    EXPECT_EQ(rightAnswer, value);
}

TEST(TReadVarIntTest, Overflow)
{
    TString input("\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x01", 11);
    TStringInput inputStream(input);
    ui64 value;
    EXPECT_ANY_THROW(ReadVarUint64(&inputStream, &value));
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
    for (int i = 0; i < numberOfValues; ++i) {
        i32 expected = static_cast<i32>(RandomNumber<ui32>());
        WriteVarInt32(&stream, expected);
        i32 actual;
        ReadVarInt32(&stream, &actual);
        EXPECT_EQ(expected, actual)
            << "Encoded Variant: " << EscapeC(stream.Str());
    }
}

////////////////////////////////////////////////////////////////////////////////

TEST(TVarInt64Test, RandomValues)
{
    srand(100500); // Set seed
    const int numberOfValues = 10000;

    TStringStream stream;
    for (int i = 0; i < numberOfValues; ++i) {
        i64 expected = static_cast<i64>(RandomNumber<ui64>());
        WriteVarInt64(&stream, expected);
        i64 actual;
        ReadVarInt64(&stream, &actual);
        EXPECT_EQ(expected, actual)
            << "Encoded Variant: " << EscapeC(stream.Str());
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
