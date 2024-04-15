#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/function_view.h>

#include <util/generic/string.h>
#include <util/string/cast.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

struct TNoCopy
{
    int Value = 42;

    TNoCopy() = default;

    TNoCopy(const TNoCopy&) = delete;

    TNoCopy(TNoCopy&&)
    { }
};

int Foo()
{
    return 42;
}

int& Bar()
{
    static int bar = 0;
    return bar;
}

const TNoCopy& ImmutBar()
{
    static TNoCopy bar = {};
    return bar;
}

TString Baz(const int& x)
{
    return ToString(x);
}

void NoExFoo() noexcept
{ }

struct TCallable
{
    int InvocationCount = 0;
    mutable int ConstInvocationCount = 0;

    void operator()() &
    {
        ++InvocationCount;
    }

    void operator()() const &
    {
        ++ConstInvocationCount;
    }
};

////////////////////////////////////////////////////////////////////////////////

TEST(TFunctionViewTest, JustWorks)
{
    auto stackLambda = [] (int val) {
        return val + 1;
    };

    {
        TFunctionView<int(int)> view(stackLambda);

        EXPECT_EQ(view(42), 43);
    }
}

TEST(TFunctionViewTest, FreeFunction)
{
    TFunctionView<int()> view(Foo);
    EXPECT_EQ(view(), 42);
}

TEST(TFunctionViewTest, RefReturn)
{
    TFunctionView<int&()> view(Bar);
    ++view();
    EXPECT_EQ(view(), 1);

    TFunctionView<const TNoCopy&()> immut_view(ImmutBar);
    EXPECT_EQ(immut_view().Value, 42);
}

TEST(TFunctionViewTest, RefArgument)
{
    TFunctionView<TString(const int&)> view(Baz);
    EXPECT_EQ(view(77), TString("77"));
}

TEST(TFunctionViewTest, NoExcept)
{
    TFunctionView<void() noexcept> view(NoExFoo);
    static_assert(std::is_nothrow_invocable_r_v<void, decltype(view)>);

    view();
}

TEST(TFunctionViewTest, CVOverloads)
{
    TCallable callable;

    TFunctionView<void()> view(callable);
    // NB: & overload overshadows every other overload.
    // const auto& viewRef = view;
    // viewRef();

    view();
    EXPECT_EQ(callable.InvocationCount, 1);
    EXPECT_EQ(callable.ConstInvocationCount, 0);
}

TEST(TFunctionViewTest, CopyView)
{
    int counter = 0;
    auto lambda = [&counter] {
        ++counter;
    };

    TFunctionView<void()> view1(lambda);
    TFunctionView<void()> view2 = view1;

    view1();
    EXPECT_EQ(counter, 1);
    view2();
    EXPECT_EQ(counter, 2);
    view1();
    EXPECT_EQ(counter, 3);
}

TEST(TFunctionViewTest, AssignView)
{
    int counter = 0;
    auto lambda = [&counter] {
        ++counter;
    };

    TFunctionView<void()> view(lambda);
    view();
    EXPECT_EQ(counter, 1);

    {
        auto innerCounter = 0;
        auto lambda = [&innerCounter] {
            ++innerCounter;
        };

        view = lambda;
        view();
        EXPECT_EQ(counter, 1);
        EXPECT_EQ(innerCounter, 1);
    }

    // NB: Even though object is dead view will remain "valid".
    // Be careful with lifetimes!
    EXPECT_TRUE(view.IsValid());
}

TEST(TFunctionViewTest, ReleaseSemantics)
{
    int counter = 0;
    auto lambda = [&counter] {
        ++counter;
    };

    TFunctionView<void()> view1(lambda);
    view1();
    EXPECT_EQ(counter, 1);

    TFunctionView view2 = view1.Release();
    EXPECT_FALSE(view1.IsValid());

    EXPECT_TRUE(view2.IsValid());

    view2();
    EXPECT_EQ(counter, 2);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
