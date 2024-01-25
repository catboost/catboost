#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/misc/enum_indexed_array.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

DEFINE_ENUM(EColor,
    ((Red)  (10))
    ((Green)(20))
    ((Blue) (30))
);

TEST(TEnumIndexedArrayTest, Size)
{
    TEnumIndexedArray<EColor, int> arr;
    EXPECT_EQ(std::ssize(arr), 21);
}

TEST(TEnumIndexedArrayTest, IsValidIndex)
{
    TEnumIndexedArray<EColor, int> arr;
    EXPECT_TRUE(arr.IsValidIndex(EColor::Red));
    EXPECT_TRUE(arr.IsValidIndex(EColor::Green));
    EXPECT_TRUE(arr.IsValidIndex(EColor::Blue));
    EXPECT_TRUE(arr.IsValidIndex(static_cast<EColor>(11)));
    EXPECT_FALSE(arr.IsValidIndex(static_cast<EColor>(9)));
}

TEST(TEnumIndexedArrayTest, Simple)
{
    TEnumIndexedArray<EColor, int> arr;
    EXPECT_EQ(arr[EColor::Red], 0);
    arr[EColor::Red] = 1;
    EXPECT_EQ(arr[EColor::Red], 1);
    EXPECT_EQ(arr[EColor::Blue], 0);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT

