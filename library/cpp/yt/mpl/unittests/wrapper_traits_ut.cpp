#include <library/cpp/yt/mpl/wrapper_traits.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <optional>
#include <string>

namespace NYT::NMpl {
namespace {

////////////////////////////////////////////////////////////////////////////////

using TNestedOptional = std::optional<std::optional<int>>;

////////////////////////////////////////////////////////////////////////////////

TEST(TWrapperTraitsTest, TrivialWrapper)
{
    using TTraits = TWrapperTraits<int>;

    static_assert(TTraits::IsTrivialWrapper);
    static_assert(std::is_same_v<TTraits::TUnwrapped, int>);
    static_assert(std::is_same_v<TTraits::TRecursiveUnwrapped, int>);
    static_assert(!CNonTrivialWrapper<int>);

    EXPECT_TRUE(TTraits::HasValue(42));
    EXPECT_TRUE(TTraits::RecursiveHasValue(42));
    EXPECT_EQ(42, TTraits::Unwrap(42));
    EXPECT_EQ(42, TTraits::Wrap(42));
}

TEST(TWrapperTraitsTest, Optional)
{
    using TTraits = TWrapperTraits<std::optional<int>>;

    static_assert(!TTraits::IsTrivialWrapper);
    static_assert(std::is_same_v<TTraits::TUnwrapped, int>);
    static_assert(std::is_same_v<TTraits::TRecursiveUnwrapped, int>);
    static_assert(CNonTrivialWrapper<std::optional<int>>);

    EXPECT_FALSE(TTraits::HasValue(std::nullopt));
    EXPECT_TRUE(TTraits::HasValue(std::optional<int>(42)));

    EXPECT_EQ(42, TTraits::Unwrap(std::optional<int>(42)));
    EXPECT_EQ(std::optional<int>(42), TTraits::Wrap(42));
}

TEST(TWrapperTraitsTest, NestedOptional)
{
    using TTraits = TWrapperTraits<TNestedOptional>;

    static_assert(std::is_same_v<TTraits::TUnwrapped, std::optional<int>>);
    static_assert(std::is_same_v<TTraits::TRecursiveUnwrapped, int>);

    EXPECT_EQ(42, TTraits::RecursiveUnwrap(TNestedOptional(std::optional<int>(42))));
    EXPECT_EQ(TNestedOptional(std::optional<int>(42)), TTraits::RecursiveWrap(42));

    EXPECT_FALSE(TTraits::RecursiveHasValue(std::nullopt));
    EXPECT_FALSE(TTraits::RecursiveHasValue(TNestedOptional(std::optional<int>())));
    EXPECT_TRUE(TTraits::RecursiveHasValue(TNestedOptional(std::optional<int>(42))));
}

TEST(TWrapperTraitsTest, NonCopyableUnwrapped)
{
    using TTraits = TWrapperTraits<std::optional<std::string>>;

    static_assert(std::is_same_v<TTraits::TUnwrapped, std::string>);

    auto wrapper = std::optional<std::string>("hello");
    EXPECT_EQ("hello", TTraits::Unwrap(std::move(wrapper)));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT::NMpl
