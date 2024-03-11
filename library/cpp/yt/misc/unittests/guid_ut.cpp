#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/misc/guid.h>

#include <util/generic//algorithm.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TGuidTest, CreateRandom)
{
    auto guid = TGuid::Create();
    auto otherGuid = TGuid::Create();
    EXPECT_FALSE(guid == otherGuid);
}

TEST(TGuidTest, FormattableGuid)
{
    EXPECT_EQ(TFormattableGuid(TGuid::FromString("1-2-3-4")).ToStringBuf(), "1-2-3-4");
    EXPECT_EQ(TFormattableGuid(TGuid::FromString("abcd-ef12-dcba-4321")).ToStringBuf(), "abcd-ef12-dcba-4321");
}

TEST(TGuidTest, StarshipOperator)
{
    std::vector<TGuid> guids{
        TGuid::FromString("abcd-ef12-dcba-4321"),
        TGuid::FromString("1-2-3-4"),
        TGuid::FromString("1-2-3-4"),
        TGuid::FromString("bada-13aa-abab-ffff"),
    };
    Sort(guids);
    for (int index = 0; index < std::ssize(guids) - 1; index++) {
        auto first = guids[index];
        auto second = guids[index];
        if (first == second) {
            EXPECT_EQ(first <=> second, std::strong_ordering::equal);
        } else {
            EXPECT_EQ(first <=> second, std::strong_ordering::less);
            EXPECT_EQ(second <=> first, std::strong_ordering::greater);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
