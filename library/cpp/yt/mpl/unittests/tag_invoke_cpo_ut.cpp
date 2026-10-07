#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/mpl/tag_invoke.h>
#include <library/cpp/yt/mpl/tag_invoke_cpo.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

inline constexpr struct TFooFn
{

    // Customizable overload.
    template <class... TArgs>
        requires NMpl::CTagInvocable<TFooFn, TArgs...>
    constexpr decltype(auto) operator() (TArgs&&... args) const
        noexcept(noexcept(NYT::NMpl::TagInvoke(*this, std::forward<TArgs>(args)...)))
    {
        return NYT::NMpl::TagInvoke(*this, std::forward<TArgs>(args)...);
    }

    // Default overload.
    template <class... TArgs>
        requires (!NMpl::CTagInvocable<TFooFn, TArgs...>)
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

    friend int TagInvoke(NMpl::TTagInvokeTag<Foo>, TCustomFoo f) noexcept(NoExcept)
    {
        return f.Val + 11;
    }
};

////////////////////////////////////////////////////////////////////////////////

TEST(TTagInvokeUsageTests, CustomOverload)
{
    static_assert(NMpl::CTagInvocable<NMpl::TTagInvokeTag<Foo>, TCustomFoo<true>>);
    static_assert(NMpl::CTagInvocable<NMpl::TTagInvokeTag<Foo>, TCustomFoo<false>>);
    static_assert(NMpl::CNothrowTagInvocable<NMpl::TTagInvokeTag<Foo>, TCustomFoo<true>>);
    static_assert(!NMpl::CNothrowTagInvocable<NMpl::TTagInvokeTag<Foo>, TCustomFoo<false>>);

    EXPECT_EQ(Foo(TCustomFoo<true>{.Val = 42}), 53);
    EXPECT_EQ(Foo(TCustomFoo<false>{.Val = 42}), 53);
}

////////////////////////////////////////////////////////////////////////////////

inline constexpr struct TBarFn
    : public NMpl::TTagInvokeCpoBase<TBarFn>
{ } Bar = {};

template <class T>
concept CBarable = requires (T&& t) {
    Bar(t);
};

////////////////////////////////////////////////////////////////////////////////

struct THasCustom
{
    friend int TagInvoke(NMpl::TTagInvokeTag<Bar>, THasCustom)
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
