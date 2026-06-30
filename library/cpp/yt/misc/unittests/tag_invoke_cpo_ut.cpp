#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/misc/tag_invoke.h>
#include <library/cpp/yt/misc/tag_invoke_cpo.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

inline constexpr struct TFooFn
{

    // Customizable overload.
    template <class... TArgs>
        requires CTagInvocable<TFooFn, TArgs...>
    constexpr decltype(auto) operator() (TArgs&&... args) const
        noexcept(noexcept(NYT::TagInvoke(*this, std::forward<TArgs>(args)...)))
    {
        return NYT::TagInvoke(*this, std::forward<TArgs>(args)...);
    }

    // Default overload.
    template <class... TArgs>
        requires (!CTagInvocable<TFooFn, TArgs...>)
    constexpr decltype(auto) operator() (TArgs&&...) const
        noexcept
    {
        return 42;
    }
} Foo = {};

////////////////////////////////////////////////////////////////////////////////

TEST(TTagInvokeUsageTests, DefaultOverload)
{
    EXPECT_EQ(Foo(42), 42);

    struct TTTT
    { };

    EXPECT_EQ(Foo(TTTT{}), 42);
}

////////////////////////////////////////////////////////////////////////////////

template <bool NoExcept>
struct TCustomFoo
{
    int Val;

    friend int TagInvoke(TTagInvokeTag<Foo>, TCustomFoo f) noexcept(NoExcept)
    {
        return f.Val + 11;
    }
};

////////////////////////////////////////////////////////////////////////////////

TEST(TTagInvokeUsageTests, CustomOverload)
{
    static_assert(CTagInvocable<TTagInvokeTag<Foo>, TCustomFoo<true>>);
    static_assert(CTagInvocable<TTagInvokeTag<Foo>, TCustomFoo<false>>);
    static_assert(CNothrowTagInvocable<TTagInvokeTag<Foo>, TCustomFoo<true>>);
    static_assert(!CNothrowTagInvocable<TTagInvokeTag<Foo>, TCustomFoo<false>>);

    EXPECT_EQ(Foo(TCustomFoo<true>{.Val = 42}), 53);
    EXPECT_EQ(Foo(TCustomFoo<false>{.Val = 42}), 53);
}

////////////////////////////////////////////////////////////////////////////////

inline constexpr struct TBarFn
    : public TTagInvokeCpoBase<TBarFn>
{ } Bar = {};

template <class T>
concept CBarable = requires (T&& t) {
    Bar(t);
};

////////////////////////////////////////////////////////////////////////////////

struct THasCustom
{
    friend int TagInvoke(TTagInvokeTag<Bar>, THasCustom)
    {
        return 11;
    }
};

////////////////////////////////////////////////////////////////////////////////

TEST(TTagInvokeCpoTests, JustWorks)
{
    struct TNoCustom
    { };
    static_assert(!CBarable<TNoCustom>);

    static_assert(CBarable<THasCustom>);
    EXPECT_EQ(Bar(THasCustom{}), 11);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
