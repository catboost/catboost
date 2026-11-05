#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/string/stream.h>

#include <util/stream/str.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TStdStringOutputTest, AppendsToReferencedString)
{
    std::string buffer = "abc";
    {
        TStdStringOutput output(buffer);
        output << "de" << 'f' << 42;
    }
    EXPECT_EQ(buffer, "abcdef42");
}

TEST(TStdStringOutputTest, MatchesTStringOutput)
{
    std::string stdBuffer;
    TString utilBuffer;
    TStdStringOutput stdOutput(stdBuffer);
    TStringOutput utilOutput(utilBuffer);
    for (int i = 0; i < 1000; ++i) {
        stdOutput << i << ',';
        utilOutput << i << ',';
    }
    EXPECT_EQ(stdBuffer, std::string(utilBuffer));
}

TEST(TStdStringStreamTest, WriteAndStr)
{
    TStdStringStream stream;
    EXPECT_FALSE(static_cast<bool>(stream));
    stream << "value=" << 123;
    EXPECT_TRUE(static_cast<bool>(stream));
    EXPECT_EQ(stream.Str(), "value=123");
    EXPECT_EQ(stream.Size(), 9u);
}

TEST(TStdStringStreamTest, ClearAndReuse)
{
    TStdStringStream stream("seed");
    EXPECT_EQ(stream.Str(), "seed");
    stream.Clear();
    EXPECT_TRUE(stream.Empty());
    stream << "again";
    EXPECT_EQ(stream.Str(), "again");
}

TEST(TStdStringStreamTest, MoveOutBuffer)
{
    TStdStringStream stream;
    stream << "movable";
    std::string extracted = std::move(stream).Str();
    EXPECT_EQ(extracted, "movable");
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
