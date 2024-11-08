////////////////////////////////////////////////////////////////////////////////
// NB(arkady-e1ppa): This decl/def order is intentional to simulate some tricky
// patterns of inclusion order.

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

// This overload should be visible by direct call
// from namespaces enclosing ns NYT
// but invisible to any template code
// therefore it must not affect CTagInvocable concept.
template <class T, class U>
int TagInvoke(const T&, const U&)
{
    return 42;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////

// NB(arkady-e1ppa): Weird name to avoid ODR violation while keeping the
// struct inside of ns NYT (and not some anonymous ones).
struct TUniquelyTaggedForTagInvokeImplUt
{
    friend int TagInvoke(TUniquelyTaggedForTagInvokeImplUt, int v)
    {
        return v + 2;
    }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

////////////////////////////////////////////////////////////////////////////////

// Actual includes start.
#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/misc/tag_invoke.h>

////////////////////////////////////////////////////////////////////////////////

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TTagInvokeCompileTests, UnqualidTemplate)
{
    static_assert(!CTagInvocable<int, int>);
    // Unqualified finds overload defined above
    // and never the CPO since constraints fail.
    EXPECT_EQ(TagInvoke(42, 42), 42);
}

TEST(TTagInvokeCompileTests, HiddenFriend)
{
    static_assert(CTagInvocable<TUniquelyTaggedForTagInvokeImplUt, int>);
    EXPECT_EQ(TagInvoke(TUniquelyTaggedForTagInvokeImplUt{}, 42), 44);
    EXPECT_EQ(NYT::TagInvoke(TUniquelyTaggedForTagInvokeImplUt{}, 42), 44);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
