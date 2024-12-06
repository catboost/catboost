#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/non_null_ptr.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

enum class EFuncResult
{
    Base,
    ConstBase,
    Derived,
    ConstDerived,
};

enum class EBinaryFuncResult
{
    OK,
    NotOK,
};

struct TBase
{ };

struct TDerived
    : public TBase
{ };

EFuncResult Foo(TNonNullPtr<TBase> /*base*/)
{
    return EFuncResult::Base;
}

EFuncResult Foo(TNonNullPtr<const TBase> /*base*/)
{
    return EFuncResult::ConstBase;
}

EFuncResult Foo(TNonNullPtr<TDerived> /*derived*/)
{
    return EFuncResult::Derived;
}

EFuncResult Foo(TNonNullPtr<const TDerived> /*derived*/)
{
    return EFuncResult::ConstDerived;
}

[[maybe_unused]] EBinaryFuncResult Foo(int* /*derived*/)
{
    return EBinaryFuncResult::NotOK;
}

EBinaryFuncResult Foo(TNonNullPtr<int> /*derived*/)
{
    return EBinaryFuncResult::OK;
}

EBinaryFuncResult Bar(TNonNullPtr<const int> /*arg*/)
{
    return EBinaryFuncResult::OK;
}

EBinaryFuncResult Baz(TNonNullPtr<int> /*arg*/)
{
    return EBinaryFuncResult::OK;
}

TEST(TNonNullPtrTest, Simple)
{
    TDerived derived{};
    const auto& constDerived = derived;
    EXPECT_EQ(EFuncResult::Derived, Foo(GetPtr(derived)));
    EXPECT_EQ(EFuncResult::ConstDerived, Foo(GetPtr(constDerived)));

    TBase base{};
    const auto& constBase = base;
    EXPECT_EQ(EFuncResult::Base, Foo(GetPtr(base)));
    EXPECT_EQ(EFuncResult::ConstBase, Foo(GetPtr(constBase)));

    int i{};
    EXPECT_EQ(EBinaryFuncResult::OK, Foo(GetPtr(i)));
}

TEST(TNonNullPtrTest, CastToConst)
{
    int i{};

    EXPECT_EQ(EBinaryFuncResult::OK, Bar(GetPtr(i)));
}

TEST(TNonNullPtrTest, ConstructionFromRawPointer)
{
    int i{};

    EXPECT_EQ(EBinaryFuncResult::OK, Baz(&i));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
