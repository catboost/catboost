#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/type_erasure.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

struct TTestCpo1
    : public TTagInvokeCpoBase<TTestCpo1>
{ };

inline constexpr TTestCpo1 TestCpo = {};

struct TCustomized
{
    int Value = 42;

    friend int TagInvoke(TTagInvokeTag<TestCpo>, const TCustomized& this_)
    {
        return this_.Value + 1;
    }
};

////////////////////////////////////////////////////////////////////////////////

// 2 Cpos won't trigger static vtable
static_assert(
    sizeof(
        TAnyRef<
            TOverload<
                TestCpo,
                void(TErasedThis&)>,
            TOverload<
                TestCpo,
                void(TErasedThis&&)>>)
    == 8 + 2 * 8);

// 3 Cpos trigger static vtable
static_assert(
    sizeof(
        TAnyRef<
            TOverload<
                TestCpo,
                void(TErasedThis&)>,
            TOverload<
                TestCpo,
                void(TErasedThis&&)>,
            TOverload<
                TestCpo,
                void(TErasedThis, int)>>)
    == 8 + 8);

////////////////////////////////////////////////////////////////////////////////

TEST(TAnyRefTest, JustWorks)
{
    using TAnyRef = TAnyRef<TOverload<TestCpo, int(const TErasedThis&)>>;
    TCustomized concrete{.Value = 42};

    TAnyRef any{concrete};
    EXPECT_EQ(TestCpo(any), 43);

    auto copy = any;
    EXPECT_EQ(TestCpo(copy), 43);

    auto movedOut = std::move(any);
    EXPECT_EQ(TestCpo(movedOut), 43);
}

TEST(TAnyRefTest, EmptyRef)
{
    TCustomized concrete{.Value = 11};

    TAnyRef<> any{concrete};

    static_assert(!std::invocable<TTagInvokeTag<TestCpo>, decltype(any)>);
    const auto& conc = any.AnyCast<TCustomized>();
    EXPECT_EQ(conc.Value, 11);
}

////////////////////////////////////////////////////////////////////////////////

struct TNoCopy
{
    TNoCopy() = default;

    TNoCopy(const TNoCopy&) = delete;

    TNoCopy(TNoCopy&&)
    { }

    TNoCopy& operator=(TNoCopy&&)
    {
        return *this;
    }

    int Val = 123;
};

static_assert(std::movable<TNoCopy>);

struct TCustomized2
{
    int Value = 1;
    static inline int DtorCount = 0;

    TCustomized2() = default;

    TCustomized2(const TCustomized2&)
    { }

    TCustomized2& operator=(const TCustomized2&)
    {
        return *this;
    }

    TCustomized2(TCustomized2&& other)
        : Value(other.Value)
    {
        other.Value = -1;
    }

    TCustomized2& operator=(TCustomized2&& other)
    {
        if (this == &other) {
            return *this;
        }

        Value = std::exchange(other.Value, -1);
        return *this;
    }

    ~TCustomized2()
    {
        ++DtorCount;
    }

    friend const TNoCopy& TagInvoke(TTagInvokeTag<TestCpo>, TCustomized2&)
    {
        static TNoCopy noCp;
        noCp.Val = 11;
        return noCp;
    }

    friend int TagInvoke(TTagInvokeTag<TestCpo>, TCustomized2&& this_)
    {
        auto v = std::move(this_);

        return 1212;
    }

    friend int TagInvoke(TTagInvokeTag<TestCpo>, TCustomized2&, int)
    {
        return 42;
    }
};

static_assert(std::copyable<TCustomized2>);

////////////////////////////////////////////////////////////////////////////////

TEST(TAnyRefTest, CvRefCorrectness)
{
    using TAnyRef = TAnyRef<
        TOverload<TestCpo, const TNoCopy&(TErasedThis&)>,
        TOverload<TestCpo, int(TErasedThis&&)>>;

    TCustomized2 cust = {};

    cust.DtorCount = 0;
    EXPECT_EQ(cust.Value, 1);

    {
        TAnyRef ref{cust};

        EXPECT_EQ(TestCpo(ref).Val, 11);
    }

    EXPECT_EQ(cust.DtorCount, 0);
    EXPECT_EQ(cust.Value, 1);

    {
        TAnyRef ref(cust);

        EXPECT_EQ(TestCpo(std::move(ref)), 1212);
    }

    EXPECT_EQ(cust.DtorCount, 1);
    EXPECT_EQ(cust.Value, -1);

    cust.Value = 1;
    TAnyRef any{cust};
    {
        TAnyRef copy{any};
        TAnyRef movedOut{std::move(any)};
        EXPECT_EQ(TestCpo(copy).Val, 11);
        EXPECT_EQ(TestCpo(movedOut).Val, 11);
    }

    EXPECT_EQ(cust.DtorCount, 1);
    EXPECT_EQ(cust.Value, 1);
}

TEST(TAnyRefTest, StaticVTableForAnyRef)
{
    using TAnyRef = TAnyRef<
        TOverload<TestCpo, const TNoCopy&(TErasedThis&)>,
        TOverload<TestCpo, int(TErasedThis&&)>,
        TOverload<TestCpo, int(TErasedThis&, int)>
    >;

    TCustomized2 cst = {};
    cst.Value = 1111;
    cst.DtorCount = 0;

    TAnyRef any{cst};
    {
        TAnyRef copy{any};
        TAnyRef movedOut{std::move(any)};
        EXPECT_EQ(TestCpo(copy).Val, 11);
        EXPECT_EQ(TestCpo(movedOut).Val, 11);
    }
    EXPECT_EQ(cst.Value, 1111);
    EXPECT_EQ(cst.DtorCount, 0);
}

////////////////////////////////////////////////////////////////////////////////

TEST(TAnyObjectTest, JustWorks)
{
    TCustomized concrete{.Value = 42};

    TAnyObject<TOverload<TestCpo, int(const TErasedThis&)>> any{concrete};
    EXPECT_EQ(TestCpo(any), 43);

    auto copy = any;
    EXPECT_EQ(TestCpo(copy), 43);

    auto movedOut = std::move(any);
    EXPECT_EQ(TestCpo(movedOut), 43);
}

TEST(TAnyObjectTest, EmptyAny)
{
    TCustomized concrete{.Value = 11};

    TAnyObject<> any{concrete};

    static_assert(!std::invocable<TTagInvokeTag<TestCpo>, decltype(any)>);
    const auto& conc = any.AnyCast<TCustomized>();
    EXPECT_EQ(conc.Value, 11);
}

TEST(TAnyObjectTest, CvRefCorrectness)
{
    using TAnyObject = TAnyObject<
        TOverload<TestCpo, const TNoCopy&(TErasedThis&)>,
        TOverload<TestCpo, int(TErasedThis&&)>>;

    TCustomized2 cust = {};

    cust.DtorCount = 0;
    EXPECT_EQ(cust.Value, 1);

    {
        TAnyObject any{cust};

        EXPECT_EQ(TestCpo(any).Val, 11);
    }

    // Any object itself.
    EXPECT_EQ(cust.DtorCount, 1);
    EXPECT_EQ(cust.Value, 1);

    {
        TAnyObject any(cust);

        EXPECT_EQ(TestCpo(std::move(any)), 1212);
    }

    // Second any object + moved out object.
    EXPECT_EQ(cust.DtorCount, 3);
    EXPECT_EQ(cust.Value, 1);

    TAnyObject any{cust};
    {
        TAnyObject copy{any};
        TAnyObject movedOut{std::move(any)};
        EXPECT_EQ(TestCpo(copy).Val, 11);
        EXPECT_EQ(TestCpo(movedOut).Val, 11);
    }

    // NB(arkady-e1ppa): Moved out any should be
    // actually empty thus moving out both moves object out
    // and destroys the moved out object.
    EXPECT_EQ(cust.DtorCount, 6);
    EXPECT_THROW(any.AnyCast<TCustomized2>(), NDetail::TBadAnyCast);
}

TEST(TAnyObjectTest, StaticVTableForAnyRef)
{
    using TAnyObject = TAnyObject<
        TOverload<TestCpo, const TNoCopy&(TErasedThis&)>,
        TOverload<TestCpo, int(TErasedThis&&)>,
        TOverload<TestCpo, int(TErasedThis&, int)>
    >;

    TCustomized2 cst = {};
    cst.Value = 1111;
    cst.DtorCount = 0;

    TAnyObject any{cst};
    {
        TAnyObject copy{any};
        TAnyObject movedOut{std::move(any)};
        EXPECT_EQ(TestCpo(copy).Val, 11);
        EXPECT_EQ(TestCpo(movedOut).Val, 11);
    }
    EXPECT_EQ(cst.Value, 1111);
    // NB(arkady-e1ppa): See comment in previous test.
    EXPECT_EQ(cst.DtorCount, 3);
    EXPECT_FALSE(any.IsValid());
}

////////////////////////////////////////////////////////////////////////////////

TEST(TAnyObjectTest, AnyMoveOnly)
{
    TAnyUnique<> any{std::in_place_type<TNoCopy>};
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
