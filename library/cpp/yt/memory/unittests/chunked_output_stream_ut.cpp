#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/chunked_output_stream.h>
#include <library/cpp/yt/memory/new.h>

#include <util/system/info.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

struct TTestStreamTag
{ };

TRefCountedTypeCookie GetTestTagCookie()
{
    return GetRefCountedTypeCookie<TTestStreamTag>();
}

// Reserve sizes are rounded up to a page, so chunk layouts are page-relative.
size_t GetTestPageSize()
{
    return NSystemInfo::GetPageSize();
}

std::string Concatenate(const std::vector<TSharedRef>& chunks)
{
    std::string result;
    for (const auto& chunk : chunks) {
        result.append(chunk.Begin(), chunk.Size());
    }
    return result;
}

std::vector<size_t> GetSizes(const std::vector<TSharedRef>& chunks)
{
    std::vector<size_t> result;
    for (const auto& chunk : chunks) {
        result.push_back(chunk.Size());
    }
    return result;
}

std::string MakePayload(size_t size)
{
    std::string result;
    result.reserve(size);
    for (size_t index = 0; index < size; ++index) {
        result.push_back(static_cast<char>('a' + index % 26));
    }
    return result;
}

void WriteByteByByte(TChunkedOutputStream* stream, TStringBuf payload)
{
    for (char ch : payload) {
        stream->Write(&ch, 1);
    }
}

////////////////////////////////////////////////////////////////////////////////

class TCountingMemoryUsageTracker
    : public ISimpleMemoryUsageTracker
{
public:
    bool Acquire(i64 size) override
    {
        Usage_ += size;
        return false;
    }

    void Release(i64 size) override
    {
        Usage_ -= size;
    }

    TSharedRef Track(TSharedRef reference, bool /*keepExistingTracking*/) override
    {
        return reference;
    }

    i64 GetUsage() const
    {
        return Usage_;
    }

private:
    i64 Usage_ = 0;
};

////////////////////////////////////////////////////////////////////////////////

TEST(TChunkedOutputStreamTest, Empty)
{
    TChunkedOutputStream stream(GetTestTagCookie());
    EXPECT_EQ(stream.GetSize(), 0u);
    EXPECT_EQ(stream.GetCapacity(), 0u);
    EXPECT_EQ(Concatenate(stream.Finish()), "");
}

TEST(TChunkedOutputStreamTest, SingleChunk)
{
    TChunkedOutputStream stream(GetTestTagCookie());
    stream.Write("hello", 5);
    EXPECT_EQ(stream.GetSize(), 5u);
    EXPECT_GE(stream.GetCapacity(), stream.GetSize());

    auto chunks = stream.Finish();
    EXPECT_EQ(GetSizes(chunks), std::vector<size_t>{5});
    EXPECT_EQ(Concatenate(chunks), "hello");
}

TEST(TChunkedOutputStreamTest, ReserveSizeDoubles)
{
    auto page = GetTestPageSize();
    TChunkedOutputStream stream(GetTestTagCookie(), GetNullSimpleMemoryUsageTracker(), page, 4 * page);

    auto payload = MakePayload(11 * page);
    WriteByteByByte(&stream, payload);
    EXPECT_EQ(stream.GetSize(), payload.size());

    auto chunks = stream.Finish();
    EXPECT_EQ(GetSizes(chunks), (std::vector<size_t>{page, 2 * page, 4 * page, 4 * page}));
    EXPECT_EQ(Concatenate(chunks), payload);
}

TEST(TChunkedOutputStreamTest, InitialReserveSizeIsCappedByMax)
{
    auto page = GetTestPageSize();
    TChunkedOutputStream stream(GetTestTagCookie(), GetNullSimpleMemoryUsageTracker(), 16 * page, page);

    auto payload = MakePayload(3 * page + page / 2);
    WriteByteByByte(&stream, payload);

    auto chunks = stream.Finish();
    EXPECT_EQ(GetSizes(chunks), (std::vector<size_t>{page, page, page, page / 2}));
    EXPECT_EQ(Concatenate(chunks), payload);
}

TEST(TChunkedOutputStreamTest, SingleWriteLargerThanMaxReserveSize)
{
    auto page = GetTestPageSize();
    TChunkedOutputStream stream(GetTestTagCookie(), GetNullSimpleMemoryUsageTracker(), page, page);

    auto payload = MakePayload(25 * page);
    stream.Write(payload.data(), payload.size());

    auto chunks = stream.Finish();
    EXPECT_EQ(GetSizes(chunks), (std::vector<size_t>{page, 24 * page}));
    EXPECT_EQ(Concatenate(chunks), payload);
}

TEST(TChunkedOutputStreamTest, PreallocateAndAdvance)
{
    auto page = GetTestPageSize();
    TChunkedOutputStream stream(GetTestTagCookie(), GetNullSimpleMemoryUsageTracker(), page, page);

    auto payload = MakePayload(10 * page);
    size_t position = 0;
    while (position < payload.size()) {
        auto size = std::min<size_t>(1000, payload.size() - position);
        auto* buffer = stream.Preallocate(size);
        ::memcpy(buffer, payload.data() + position, size);
        stream.Advance(size);
        position += size;
        EXPECT_EQ(stream.GetSize(), position);
    }

    EXPECT_EQ(Concatenate(stream.Finish()), payload);
}

TEST(TChunkedOutputStreamTest, PreallocateLargerThanChunk)
{
    auto page = GetTestPageSize();
    TChunkedOutputStream stream(GetTestTagCookie(), GetNullSimpleMemoryUsageTracker(), page, page);

    stream.Write("head", 4);

    auto payload = MakePayload(25 * page);
    auto* buffer = stream.Preallocate(payload.size());
    ::memcpy(buffer, payload.data(), payload.size());
    stream.Advance(payload.size());

    EXPECT_EQ(stream.GetSize(), 4 + payload.size());
    EXPECT_EQ(Concatenate(stream.Finish()), "head" + payload);
}

TEST(TChunkedOutputStreamTest, NextAndUndo)
{
    auto page = GetTestPageSize();
    TChunkedOutputStream stream(GetTestTagCookie(), GetNullSimpleMemoryUsageTracker(), page, page);

    void* buffer = nullptr;
    auto size = stream.Next(&buffer);
    EXPECT_EQ(size, page);
    EXPECT_EQ(stream.GetSize(), page);

    ::memset(buffer, 'x', 10);
    stream.Undo(size - 10);
    EXPECT_EQ(stream.GetSize(), 10u);
    EXPECT_EQ(Concatenate(stream.Finish()), std::string(10, 'x'));
}

TEST(TChunkedOutputStreamTest, FinishResetsStream)
{
    TChunkedOutputStream stream(GetTestTagCookie());
    stream.Write("hello", 5);
    EXPECT_EQ(Concatenate(stream.Finish()), "hello");

    EXPECT_EQ(stream.GetSize(), 0u);
    EXPECT_EQ(stream.GetCapacity(), 0u);
}

TEST(TChunkedOutputStreamTest, Move)
{
    TChunkedOutputStream stream(GetTestTagCookie());
    stream.Write("hello", 5);

    auto movedStream = std::move(stream);
    movedStream.Write("world", 5);
    EXPECT_EQ(movedStream.GetSize(), 10u);
    EXPECT_EQ(Concatenate(movedStream.Finish()), "helloworld");
}

TEST(TChunkedOutputStreamTest, MemoryUsageTracksCurrentChunk)
{
    auto page = GetTestPageSize();
    auto tracker = New<TCountingMemoryUsageTracker>();

    TChunkedOutputStream stream(GetTestTagCookie(), tracker, page, page);
    EXPECT_EQ(tracker->GetUsage(), 0);

    stream.Write("hello", 5);
    EXPECT_EQ(tracker->GetUsage(), static_cast<i64>(page));

    auto payload = MakePayload(10 * page);
    stream.Write(payload.data(), payload.size());
    EXPECT_EQ(tracker->GetUsage(), static_cast<i64>(10 * page));

    stream.Finish();
    EXPECT_EQ(tracker->GetUsage(), 0);
}

TEST(TChunkedOutputStreamTest, MemoryUsageReleasedOnDestruction)
{
    auto page = GetTestPageSize();
    auto tracker = New<TCountingMemoryUsageTracker>();
    {
        TChunkedOutputStream stream(GetTestTagCookie(), tracker, page, page);
        stream.Write("hello", 5);
        EXPECT_GT(tracker->GetUsage(), 0);
    }
    EXPECT_EQ(tracker->GetUsage(), 0);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
