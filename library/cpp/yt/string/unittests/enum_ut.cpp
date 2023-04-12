#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/string/enum.h>
#include <library/cpp/yt/string/format.h>

#include <limits>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

// Some compile-time sanity checks.
DEFINE_ENUM(ESample, (One)(Two));
static_assert(TFormatTraits<ESample>::HasCustomFormatValue);
static_assert(TFormatTraits<TEnumIndexedVector<ESample, int>>::HasCustomFormatValue);

DEFINE_ENUM(EColor,
    (Red)
    (BlackAndWhite)
);

DEFINE_BIT_ENUM(ELangs,
    ((None)       (0x00))
    ((Cpp)        (0x01))
    ((Go)         (0x02))
    ((Rust)       (0x04))
    ((Python)     (0x08))
    ((JavaScript) (0x10))
)

TEST(TFormatTest, Enum)
{
    EXPECT_EQ("Red", Format("%v", EColor::Red));
    EXPECT_EQ("red", Format("%lv", EColor::Red));

    EXPECT_EQ("BlackAndWhite", Format("%v", EColor::BlackAndWhite));
    EXPECT_EQ("black_and_white", Format("%lv", EColor::BlackAndWhite));

    EXPECT_EQ("EColor(100)", Format("%v", EColor(100)));

    EXPECT_EQ("JavaScript", Format("%v", ELangs::JavaScript));
    EXPECT_EQ("java_script", Format("%lv", ELangs::JavaScript));

    EXPECT_EQ("None", Format("%v", ELangs::None));
    EXPECT_EQ("none", Format("%lv", ELangs::None));

    EXPECT_EQ("Cpp | Go", Format("%v", ELangs::Cpp | ELangs::Go));
    EXPECT_EQ("cpp | go", Format("%lv", ELangs::Cpp | ELangs::Go));

    auto four = ELangs::Cpp | ELangs::Go | ELangs::Python | ELangs::JavaScript;
    EXPECT_EQ("Cpp | Go | Python | JavaScript", Format("%v", four));
    EXPECT_EQ("cpp | go | python | java_script", Format("%lv", four));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT


