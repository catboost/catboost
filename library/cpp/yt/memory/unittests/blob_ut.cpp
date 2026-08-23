#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/blob.h>

#include <util/system/info.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

struct TTestBlobTag
{ };

TBlob MakeBlob(TStringBuf data)
{
    TBlob blob(GetRefCountedTypeCookie<TTestBlobTag>());
    blob.Append(data.data(), data.size());
    return blob;
}

////////////////////////////////////////////////////////////////////////////////

TEST(TBlobTest, DefaultConstructed)
{
    TBlob blob;
    EXPECT_TRUE(blob.IsEmpty());
    EXPECT_EQ(blob.Size(), 0u);
    EXPECT_EQ(blob.size(), 0u);
    EXPECT_EQ(blob.Capacity(), 0u);
    EXPECT_EQ(blob.Begin(), blob.End());
    EXPECT_EQ(blob.ToStringBuf(), "");
}

TEST(TBlobTest, ConstructedWithSize)
{
    TBlob blob(GetRefCountedTypeCookie<TTestBlobTag>(), /*size*/ 10);
    EXPECT_FALSE(blob.IsEmpty());
    EXPECT_EQ(blob.Size(), 10u);
    EXPECT_GE(blob.Capacity(), 10u);
    EXPECT_EQ(blob.End() - blob.Begin(), 10);
    for (size_t index = 0; index < blob.Size(); ++index) {
        EXPECT_EQ(blob[index], 0);
    }
}

TEST(TBlobTest, ConstructedFromRef)
{
    TStringBuf data = "hello";
    TBlob blob(GetRefCountedTypeCookie<TTestBlobTag>(), TRef::FromStringBuf(data));
    EXPECT_EQ(blob.ToStringBuf(), data);
    EXPECT_NE(blob.Begin(), data.data());
}

TEST(TBlobTest, Append)
{
    TBlob blob(GetRefCountedTypeCookie<TTestBlobTag>());
    blob.Append(TRef::FromStringBuf("ab"));
    EXPECT_EQ(blob.ToStringBuf(), "ab");

    blob.Append("cd", 2);
    EXPECT_EQ(blob.ToStringBuf(), "abcd");

    blob.Append('e');
    EXPECT_EQ(blob.ToStringBuf(), "abcde");
    EXPECT_EQ(blob.Size(), 5u);
}

TEST(TBlobTest, AppendGrowth)
{
    TBlob blob(GetRefCountedTypeCookie<TTestBlobTag>());
    std::string expected;
    for (int index = 0; index < 10'000; ++index) {
        char ch = 'a' + index % 26;
        blob.Append(ch);
        expected.push_back(ch);
        ASSERT_GE(blob.Capacity(), blob.Size());
    }
    EXPECT_EQ(blob.ToStringBuf(), expected);
}

TEST(TBlobTest, Resize)
{
    auto blob = MakeBlob("abcde");

    blob.Resize(3);
    EXPECT_EQ(blob.ToStringBuf(), "abc");

    blob.Resize(5);
    EXPECT_EQ(blob.ToStringBuf(), TStringBuf("abc\0\0", 5));
}

TEST(TBlobTest, ResizeKeepsCapacity)
{
    auto blob = MakeBlob("abcde");
    auto capacity = blob.Capacity();

    blob.Resize(0);
    EXPECT_TRUE(blob.IsEmpty());
    EXPECT_EQ(blob.Capacity(), capacity);
}

TEST(TBlobTest, Reserve)
{
    auto blob = MakeBlob("abc");

    blob.Reserve(1000);
    EXPECT_GE(blob.Capacity(), 1000u);
    EXPECT_EQ(blob.Size(), 3u);
    EXPECT_EQ(blob.ToStringBuf(), "abc");

    auto capacity = blob.Capacity();
    blob.Reserve(1);
    EXPECT_EQ(blob.Capacity(), capacity);
}

TEST(TBlobTest, Clear)
{
    auto blob = MakeBlob("abc");
    auto capacity = blob.Capacity();

    blob.Clear();
    EXPECT_TRUE(blob.IsEmpty());
    EXPECT_EQ(blob.Capacity(), capacity);

    blob.Append('x');
    EXPECT_EQ(blob.ToStringBuf(), "x");
}

TEST(TBlobTest, Copy)
{
    auto blob = MakeBlob("abc");

    TBlob copy(blob);
    EXPECT_EQ(copy.ToStringBuf(), "abc");
    EXPECT_NE(copy.Begin(), blob.Begin());

    copy[0] = 'z';
    EXPECT_EQ(blob.ToStringBuf(), "abc");
}

TEST(TBlobTest, CopyAssign)
{
    auto blob = MakeBlob("abc");
    auto copy = MakeBlob("0123456789");

    copy = blob;
    EXPECT_EQ(copy.ToStringBuf(), "abc");
    EXPECT_NE(copy.Begin(), blob.Begin());
}

TEST(TBlobTest, CopyEmpty)
{
    TBlob blob;
    TBlob copy(blob);
    EXPECT_TRUE(copy.IsEmpty());
    EXPECT_EQ(copy.Capacity(), 0u);
}

TEST(TBlobTest, Move)
{
    auto blob = MakeBlob("abc");
    auto* begin = blob.Begin();

    TBlob moved(std::move(blob));
    EXPECT_EQ(moved.ToStringBuf(), "abc");
    EXPECT_EQ(moved.Begin(), begin);
    EXPECT_TRUE(blob.IsEmpty());
    EXPECT_EQ(blob.Capacity(), 0u);
    EXPECT_EQ(blob.Begin(), nullptr);
}

TEST(TBlobTest, MoveAssign)
{
    auto blob = MakeBlob("abc");
    auto other = MakeBlob("0123456789");

    other = std::move(blob);
    EXPECT_EQ(other.ToStringBuf(), "abc");
    EXPECT_TRUE(blob.IsEmpty());
}

TEST(TBlobTest, Swap)
{
    auto lhs = MakeBlob("abc");
    auto rhs = MakeBlob("0123456789");

    swap(lhs, rhs);
    EXPECT_EQ(lhs.ToStringBuf(), "0123456789");
    EXPECT_EQ(rhs.ToStringBuf(), "abc");

    swap(lhs, rhs);
    EXPECT_EQ(lhs.ToStringBuf(), "abc");
    EXPECT_EQ(rhs.ToStringBuf(), "0123456789");
}

TEST(TBlobTest, SelfSwap)
{
    auto blob = MakeBlob("abc");
    swap(blob, blob);
    EXPECT_EQ(blob.ToStringBuf(), "abc");
}

TEST(TBlobTest, ToRef)
{
    auto blob = MakeBlob("abc");
    auto ref = blob.ToRef();
    EXPECT_EQ(ref.Begin(), blob.Begin());
    EXPECT_EQ(ref.Size(), blob.Size());
}

TEST(TBlobTest, MutableAccess)
{
    auto blob = MakeBlob("abc");
    blob[1] = 'z';
    EXPECT_EQ(blob.ToStringBuf(), "azc");
    *blob.Begin() = 'y';
    EXPECT_EQ(blob.ToStringBuf(), "yzc");
}

TEST(TBlobTest, PageAligned)
{
    TBlob blob(
        GetRefCountedTypeCookie<TTestBlobTag>(),
        /*size*/ 100,
        /*initializeStorage*/ true,
        /*pageAligned*/ true);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(blob.Begin()) % NSystemInfo::GetPageSize(), 0u);

    blob.Resize(100'000);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(blob.Begin()) % NSystemInfo::GetPageSize(), 0u);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
