#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/misc/cast.h>
#include <library/cpp/yt/misc/enum.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

DEFINE_ENUM_WITH_UNDERLYING_TYPE(ECardinal, char,
    ((West)  (0))
    ((North) (1))
    ((East)  (2))
    ((South) (3))
);

DEFINE_BIT_ENUM_WITH_UNDERLYING_TYPE(EFeatures, ui8,
    ((None)  (0x0000))
    ((First) (0x0001))
    ((Second)(0x0002))
);

TEST(TCastTest, TryCheckedEnumCast)
{
    EXPECT_EQ((TryCheckedEnumCast<ECardinal, char>(2)), ECardinal::East);
    EXPECT_EQ((TryCheckedEnumCast<ECardinal, int>(3)), ECardinal::South);

    EXPECT_FALSE((TryCheckedEnumCast<ECardinal, char>(100)));
    EXPECT_FALSE((TryCheckedEnumCast<ECardinal, int>(300)));

    EXPECT_EQ((TryCheckedEnumCast<EFeatures, ui8>(0)), EFeatures::None);
    EXPECT_EQ((TryCheckedEnumCast<EFeatures, ui8>(ToUnderlying(EFeatures::First))), EFeatures::First);
    EXPECT_EQ((TryCheckedEnumCast<EFeatures, ui8>(ToUnderlying(EFeatures::Second))), EFeatures::Second);
    EXPECT_EQ((TryCheckedEnumCast<EFeatures, int>(ToUnderlying(EFeatures::First))), EFeatures::First);
    EXPECT_EQ((TryCheckedEnumCast<EFeatures, ui8>(ToUnderlying(EFeatures::First | EFeatures::Second))), EFeatures::First | EFeatures::Second);

    EXPECT_FALSE((TryCheckedEnumCast<EFeatures, ui8>(0x10)));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT

