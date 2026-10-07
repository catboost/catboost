#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/chunked_input_stream.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TSharedRef MakeBlock(TStringBuf data)
{
    return TSharedRef::FromString(std::string(data));
}

std::vector<TSharedRef> MakeBlocks(const std::vector<TStringBuf>& parts)
{
    std::vector<TSharedRef> result;
    for (auto part : parts) {
        result.push_back(MakeBlock(part));
    }
    return result;
}

std::string ReadAll(IZeroCopyInput* input)
{
    std::string result;
    const void* ptr = nullptr;
    while (auto size = input->Next(&ptr)) {
        result.append(static_cast<const char*>(ptr), size);
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////

TEST(TChunkedInputStreamTest, NoBlocks)
{
    TChunkedInputStream stream({});

    const void* ptr = nullptr;
    EXPECT_EQ(stream.Next(&ptr, 100), 0u);
    EXPECT_EQ(ptr, nullptr);
}

TEST(TChunkedInputStreamTest, SingleBlock)
{
    TChunkedInputStream stream(MakeBlocks({"hello"}));
    EXPECT_EQ(ReadAll(&stream), "hello");
}

TEST(TChunkedInputStreamTest, MultipleBlocks)
{
    TChunkedInputStream stream(MakeBlocks({"hello", " ", "world"}));
    EXPECT_EQ(ReadAll(&stream), "hello world");
}

TEST(TChunkedInputStreamTest, EmptyBlocksAreSkipped)
{
    TChunkedInputStream stream(MakeBlocks({"", "a", "", "", "b", ""}));
    EXPECT_EQ(ReadAll(&stream), "ab");
}

TEST(TChunkedInputStreamTest, AllBlocksEmpty)
{
    TChunkedInputStream stream(MakeBlocks({"", "", ""}));

    const void* ptr = nullptr;
    EXPECT_EQ(stream.Next(&ptr, 100), 0u);
    EXPECT_EQ(ptr, nullptr);
}

TEST(TChunkedInputStreamTest, NextNeverCrossesBlockBoundary)
{
    auto blocks = MakeBlocks({"abc", "de"});
    TChunkedInputStream stream(blocks);

    const void* ptr = nullptr;
    EXPECT_EQ(stream.Next(&ptr, 100), 3u);
    EXPECT_EQ(TStringBuf(static_cast<const char*>(ptr), 3), "abc");

    EXPECT_EQ(stream.Next(&ptr, 100), 2u);
    EXPECT_EQ(TStringBuf(static_cast<const char*>(ptr), 2), "de");

    EXPECT_EQ(stream.Next(&ptr, 100), 0u);
}

TEST(TChunkedInputStreamTest, NextRespectsLengthLimit)
{
    TChunkedInputStream stream(MakeBlocks({"abcde"}));

    const void* ptr = nullptr;
    EXPECT_EQ(stream.Next(&ptr, 2), 2u);
    EXPECT_EQ(TStringBuf(static_cast<const char*>(ptr), 2), "ab");

    EXPECT_EQ(stream.Next(&ptr, 2), 2u);
    EXPECT_EQ(TStringBuf(static_cast<const char*>(ptr), 2), "cd");

    EXPECT_EQ(stream.Next(&ptr, 2), 1u);
    EXPECT_EQ(TStringBuf(static_cast<const char*>(ptr), 1), "e");

    EXPECT_EQ(stream.Next(&ptr, 2), 0u);
}

TEST(TChunkedInputStreamTest, NextReturnsPointersIntoBlocks)
{
    auto blocks = MakeBlocks({"abc", "de"});
    TChunkedInputStream stream(blocks);

    const void* ptr = nullptr;
    stream.Next(&ptr, 1);
    EXPECT_EQ(ptr, blocks[0].Begin());

    stream.Next(&ptr, 100);
    EXPECT_EQ(ptr, blocks[0].Begin() + 1);
}

TEST(TChunkedInputStreamTest, Read)
{
    TChunkedInputStream stream(MakeBlocks({"hello", " ", "world"}));

    char buffer[7];
    EXPECT_EQ(stream.Load(buffer, 7), 7u);
    EXPECT_EQ(TStringBuf(buffer, 7), "hello w");
    EXPECT_EQ(stream.ReadAll(), "orld");
}

TEST(TChunkedInputStreamTest, SkipStopsAtBlockBoundary)
{
    TChunkedInputStream stream(MakeBlocks({"hello", " ", "world"}));

    EXPECT_EQ(stream.Skip(6), 5u);
    EXPECT_EQ(stream.Skip(6), 1u);
    EXPECT_EQ(stream.ReadAll(), "world");
}

TEST(TChunkedInputStreamTest, SkipPastEnd)
{
    TChunkedInputStream stream(MakeBlocks({"abc"}));

    EXPECT_EQ(stream.Skip(100), 3u);
    EXPECT_EQ(stream.ReadAll(), "");
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
