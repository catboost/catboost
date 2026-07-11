#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/testing/gtest_extensions/assertions.h>

#include <library/cpp/yt/memory/ref.h>

#include <util/generic/size_literals.h>

namespace NYT::NYson {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TSharedRefTest, Save)
{
    const TSharedRef expected = TSharedRef::FromString(std::string("My tests data"));
    TStringStream s;
    ::Save(&s, expected);  // only Save supported for TSharedRef. You can ::Load serialized data to vector.
}

////////////////////////////////////////////////////////////////////////////////

constexpr char Payload = 0x42;

TEST(TSharedMutableRefTest, Allocate)
{
    auto ref = TSharedMutableRef::Allocate(64_KB);
    EXPECT_EQ(ref.Size(), 64_KB);

    // InitializeStorage defaults to true.
    for (char ch : ref) {
        EXPECT_EQ(ch, 0);
    }

    // The whole range must be writable.
    ::memset(ref.Begin(), Payload, ref.Size());
    EXPECT_EQ(ref[0], Payload);
    EXPECT_EQ(ref[ref.Size() - 1], Payload);
}

TEST(TSharedMutableRefTest, AllocateNoInitialize)
{
    auto ref = TSharedMutableRef::Allocate(64_KB, {.InitializeStorage = false});
    EXPECT_EQ(ref.Size(), 64_KB);
    ::memset(ref.Begin(), Payload, ref.Size());
}

TEST(TSharedMutableRefTest, AllocateExtendToUsableSize)
{
    auto ref = TSharedMutableRef::Allocate(64_KB, {.ExtendToUsableSize = true});
    EXPECT_GE(ref.Size(), 64_KB);

    // InitializeStorage defaults to true.
    for (char ch : ref) {
        EXPECT_EQ(ch, 0);
    }

    ::memset(ref.Begin(), Payload, ref.Size());
    EXPECT_EQ(ref[ref.Size() - 1], Payload);
}

TEST(TSharedMutableRefTest, AllocateNoInitializeExtendToUsableSize)
{
    auto ref = TSharedMutableRef::Allocate(
        64_KB,
        {.InitializeStorage = false, .ExtendToUsableSize = true});
    EXPECT_GE(ref.Size(), 64_KB);
    ::memset(ref.Begin(), Payload, ref.Size());
}

TEST(TSharedMutableRefTest, AllocateViaMmap)
{
    auto ref = TSharedMutableRef::AllocateViaMmap(64_KB);
    EXPECT_EQ(ref.Size(), 64_KB);

    // InitializeStorage defaults to true.
    for (char ch : ref) {
        EXPECT_EQ(ch, 0);
    }

    // The whole range must be writable.
    ::memset(ref.Begin(), Payload, ref.Size());
    EXPECT_EQ(ref[0], Payload);
    EXPECT_EQ(ref[ref.Size() - 1], Payload);
}

TEST(TSharedMutableRefTest, AllocateViaMmapNoInitialize)
{
    auto ref = TSharedMutableRef::AllocateViaMmap(64_KB, {.InitializeStorage = false});
    EXPECT_EQ(ref.Size(), 64_KB);
    ::memset(ref.Begin(), Payload, ref.Size());
}

TEST(TSharedMutableRefTest, AllocateViaMmapThp)
{
    // Larger than a single huge page to exercise the rounded-up mapping.
    constexpr size_t Size = 5_MB;
    auto ref = TSharedMutableRef::AllocateViaMmap(Size, {.UseThp = true});
    EXPECT_EQ(ref.Size(), Size);
    ::memset(ref.Begin(), Payload, ref.Size());
    EXPECT_EQ(ref[Size - 1], Payload);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT::NYson
