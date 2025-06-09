#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/new.h>
#include <library/cpp/yt/memory/non_null_ptr.h>
#include <library/cpp/yt/memory/ref_counted.h>

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

class TClass
    : public TRefCounted
{
public:
    TClass() = default;
};

EBinaryFuncResult Bar(TNonNullPtr<TClass> /*arg*/)
{
    return EBinaryFuncResult::OK;
}

EBinaryFuncResult Baz(TNonNullPtr<const TClass> /*arg*/)
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

TEST(TNonNullPtrTest, ConstructionFromIntrusivePtr)
{
    TIntrusivePtr<TClass> obj = New<TClass>();
    EXPECT_EQ(EBinaryFuncResult::OK, Bar(obj));
}

TEST(TNonNullPtrTest, ConstructionConstFromIntrusivePtr)
{
    TIntrusivePtr<TClass> obj1 = New<TClass>();
    TNonNullPtr<TClass> obj1Ptr = obj1;
    EXPECT_EQ(EBinaryFuncResult::OK, Baz(obj1Ptr));

    TIntrusivePtr<const TClass> obj2 = obj1;
    EXPECT_EQ(EBinaryFuncResult::OK, Baz(obj2));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
