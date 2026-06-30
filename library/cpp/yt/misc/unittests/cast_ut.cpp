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

DEFINE_BIT_ENUM(ELangsWithUnknown,
    ((None)       (0x00))
    ((Cpp)        (0x01))
    ((Go)         (0x02))
    ((Rust)       (0x04))
    ((Python)     (0x08))
    ((JavaScript) (0x10))
    ((CppGo)      (0x03))
    ((All)        (0x1f))
    ((Unknown)    (0x20))
);
DEFINE_ENUM_UNKNOWN_VALUE(ELangsWithUnknown, Unknown);

DEFINE_ENUM(EColorWithUnknown,
    ((Red)     (0))
    ((Green)   (1))
    ((Unknown) (2))
);
DEFINE_ENUM_UNKNOWN_VALUE(EColorWithUnknown, Unknown);

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

    EXPECT_FALSE(TryCheckedEnumCast<EColorWithUnknown>(3));
    EXPECT_EQ(TryCheckedEnumCast<EColorWithUnknown>(3, /*enableUnknown*/ true), EColorWithUnknown::Unknown);

    EXPECT_FALSE(TryCheckedEnumCast<ELangsWithUnknown>(0x40));
    EXPECT_EQ(TryCheckedEnumCast<ELangsWithUnknown>(0x40, /*enableUnknown*/ true), ELangsWithUnknown::Unknown);
    EXPECT_EQ(TryCheckedEnumCast<ELangsWithUnknown>(0x41, /*enableUnknown*/ true), ELangsWithUnknown::Unknown | ELangsWithUnknown::Cpp);
}

TEST(TCastTest, CheckedEnumCast)
{
    EXPECT_EQ((CheckedEnumCast<ECardinal, char>(2)), ECardinal::East);
    EXPECT_EQ((CheckedEnumCast<ECardinal, int>(3)), ECardinal::South);

    EXPECT_THROW((CheckedEnumCast<ECardinal, char>(100)), TSimpleException);
    EXPECT_THROW((CheckedEnumCast<ECardinal, int>(300)), TSimpleException);

    EXPECT_EQ((CheckedEnumCast<EFeatures, ui8>(0)), EFeatures::None);
    EXPECT_EQ((CheckedEnumCast<EFeatures, ui8>(ToUnderlying(EFeatures::First))), EFeatures::First);
    EXPECT_EQ((CheckedEnumCast<EFeatures, ui8>(ToUnderlying(EFeatures::Second))), EFeatures::Second);
    EXPECT_EQ((CheckedEnumCast<EFeatures, int>(ToUnderlying(EFeatures::First))), EFeatures::First);
    EXPECT_EQ((CheckedEnumCast<EFeatures, ui8>(ToUnderlying(EFeatures::First | EFeatures::Second))), EFeatures::First | EFeatures::Second);

    EXPECT_THROW((CheckedEnumCast<EFeatures, ui8>(0x10)), TSimpleException);

    EXPECT_EQ(CheckedEnumCast<EColorWithUnknown>(3), EColorWithUnknown::Unknown);

    EXPECT_EQ(CheckedEnumCast<ELangsWithUnknown>(0x40), ELangsWithUnknown::Unknown);
    EXPECT_EQ(CheckedEnumCast<ELangsWithUnknown>(0x41), ELangsWithUnknown::Unknown | ELangsWithUnknown::Cpp);
}

TEST(TCastTest, IntegralCasts)
{
    static_assert(CanFitSubtype<i64, i32>());
    static_assert(CanFitSubtype<ui64, ui32>());
    static_assert(CanFitSubtype<ui64, ui64>());
    static_assert(!CanFitSubtype<ui64, i32>());
    static_assert(!CanFitSubtype<i32, i64>());

    static_assert(IsInIntegralRange<ui32>(0));
    static_assert(IsInIntegralRange<ui32>(1ull));
    static_assert(!IsInIntegralRange<ui32>(-1));
    static_assert(!IsInIntegralRange<i32>(std::numeric_limits<i64>::max()));

    static_assert(IsInIntegralRange<i32>(1ull));
    static_assert(IsInIntegralRange<i32>(-1));
    static_assert(!IsInIntegralRange<ui32>(-1));
    static_assert(!IsInIntegralRange<ui32>(std::numeric_limits<i64>::max()));
    static_assert(IsInIntegralRange<ui64>(std::numeric_limits<i64>::max()));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
