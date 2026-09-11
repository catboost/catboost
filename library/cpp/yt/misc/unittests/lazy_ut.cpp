#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/misc/lazy.h>

#include <concepts>
#include <string>
#include <utility>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

template <class... TArgs>
struct TDeducedPack
{ };

//! Models the |TArgs&&...| pack that #Force's result is forwarded into.
template <class... TArgs>
TDeducedPack<TArgs...> DeduceFrom(TArgs&&...);

template <class T>
constexpr bool ForceMatchesDeduction =
    std::same_as<decltype(DeduceFrom(Force(std::declval<T>()))), TDeducedPack<TForced<T>>>;

int& GetRef()
{
    static int Value = 7;
    return Value;
}

////////////////////////////////////////////////////////////////////////////////

TEST(TLazyTest, ForcePassesPlainValuesThrough)
{
    EXPECT_EQ(Force(123), 123);

    int value = 42;
    EXPECT_EQ(&Force(value), &value);
}

TEST(TLazyTest, ForceEvaluatesLazyValues)
{
    int calls = 0;
    auto lazy = YT_LAZY((++calls, 123));
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(Force(lazy), 123);
    EXPECT_EQ(calls, 1);
}

TEST(TLazyTest, NotMemoized)
{
    int calls = 0;
    auto lazy = YT_LAZY((++calls, 123));
    EXPECT_EQ(Force(lazy), 123);
    EXPECT_EQ(Force(lazy), 123);
    EXPECT_EQ(calls, 2);
}

TEST(TLazyTest, CapturesEnclosingScope)
{
    std::string value = "hello";
    const std::string* pointer = &value;
    EXPECT_EQ(Force(YT_LAZY(pointer->size())), 5u);
}

TEST(TLazyTest, ExpressionMayContainCommas)
{
    auto lazy = YT_LAZY(std::pair<int, int>(1, 2));
    EXPECT_EQ(Force(lazy).second, 2);
}

TEST(TLazyTest, Concept)
{
    static_assert(!CLazy<int>);

    using TLazyInt = decltype(YT_LAZY(123));
    static_assert(CLazy<TLazyInt>);
    static_assert(CLazy<TLazyInt&>);
    static_assert(CLazy<const TLazyInt&>);
    static_assert(CLazy<TLazyInt&&>);
}

TEST(TLazyTest, ForcedType)
{
    static_assert(std::same_as<TForced<int>, int>);
    static_assert(std::same_as<TForced<int&>, int&>);
    static_assert(std::same_as<TForced<const std::string&>, const std::string&>);

    static_assert(std::same_as<TForced<decltype(YT_LAZY(123))>, int>);
}

TEST(TLazyTest, ForcedTypeMatchesDeductionForEveryCategory)
{
    static_assert(ForceMatchesDeduction<int>);
    static_assert(ForceMatchesDeduction<int&>);
    static_assert(ForceMatchesDeduction<const int&>);
    static_assert(ForceMatchesDeduction<std::string>);
    static_assert(ForceMatchesDeduction<const std::string&>);

    // The one input for which the non-lazy path collapses anything.
    static_assert(std::same_as<TForced<std::string&&>, std::string>);
    static_assert(ForceMatchesDeduction<std::string&&>);

    std::string value;

    // #CLazy admits every cvref form.
    using TLazyValue = decltype(YT_LAZY(123));
    static_assert(ForceMatchesDeduction<TLazyValue>);
    static_assert(ForceMatchesDeduction<TLazyValue&>);
    static_assert(ForceMatchesDeduction<const TLazyValue&>);

    using TLazyLvalueRef = decltype(YT_LAZY(GetRef()));
    static_assert(std::same_as<TForced<TLazyLvalueRef>, int&>);
    static_assert(ForceMatchesDeduction<TLazyLvalueRef>);

    using TLazyRvalueRef = decltype(YT_LAZY(std::move(value)));
    static_assert(std::same_as<TForced<TLazyRvalueRef>, std::string>);
    static_assert(ForceMatchesDeduction<TLazyRvalueRef>);
}

TEST(TLazyTest, ValueCategoryOfOperand)
{
    std::string value;

    // Lvalues stay references, so the taken branch does not copy. NB: An operand rooting a
    // reference in a temporary is rejected by -Wreturn-stack-address, so it cannot be
    // asserted on here.
    static_assert(std::same_as<TForced<decltype(YT_LAZY(value))>, std::string&>);
    static_assert(std::same_as<TForced<decltype(YT_LAZY(std::as_const(value)))>, const std::string&>);
    static_assert(std::same_as<TForced<decltype(YT_LAZY(GetRef()))>, int&>);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
