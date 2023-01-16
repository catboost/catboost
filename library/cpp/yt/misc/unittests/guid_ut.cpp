#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/misc/guid.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TGuidTest, RandomGuids)
{
    auto guid = TGuid::Create();
    auto otherGuid = TGuid::Create();
    EXPECT_FALSE(guid == otherGuid);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
