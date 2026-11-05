#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/exact_ref_counted_cast.h>
#include <library/cpp/yt/memory/new.h>
#include <library/cpp/yt/memory/ref_counted.h>

#include <type_traits>
#include <typeinfo>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

struct TBase
    : public TRefCounted
{ };

struct TDerived
    : public TBase
{
    int Tag = 42;
};

struct TMoreDerived
    : public TDerived
{ };

////////////////////////////////////////////////////////////////////////////////
// Pointer form.

TEST(TExactRefCountedCastTest, ExactMatch)
{
    auto object = New<TDerived>();
    TBase* base = object.Get();
    EXPECT_EQ(ExactRefCountedCast<TDerived>(base), object.Get());
    EXPECT_EQ(ExactRefCountedCast<TDerived>(base)->Tag, 42);
}

TEST(TExactRefCountedCastTest, SubclassDoesNotMatch)
{
    auto object = New<TMoreDerived>();
    TBase* base = object.Get();
    // Object was New<TMoreDerived>(); an exact probe for TDerived must miss.
    EXPECT_EQ(ExactRefCountedCast<TDerived>(base), nullptr);
    // ...but the exact type matches.
    EXPECT_EQ(ExactRefCountedCast<TMoreDerived>(base), object.Get());
}

TEST(TExactRefCountedCastTest, BaseDoesNotMatch)
{
    auto object = New<TDerived>();
    TBase* base = object.Get();
    // It is exactly a TDerived, not a TBase.
    EXPECT_EQ(ExactRefCountedCast<TBase>(base), nullptr);
}

TEST(TExactRefCountedCastTest, Null)
{
    TBase* base = nullptr;
    EXPECT_EQ(ExactRefCountedCast<TDerived>(base), nullptr);
}

TEST(TExactRefCountedCastTest, ConstnessPreserved)
{
    auto object = New<TDerived>();
    const TBase* base = object.Get();
    auto* result = ExactRefCountedCast<TDerived>(base);
    static_assert(std::is_same_v<decltype(result), const TDerived*>);
    EXPECT_EQ(result, object.Get());
}

////////////////////////////////////////////////////////////////////////////////
// Reference form.

TEST(TExactRefCountedCastTest, RefExactMatch)
{
    auto object = New<TDerived>();
    TBase& base = *object;
    EXPECT_EQ(&ExactRefCountedCast<TDerived>(base), object.Get());
}

TEST(TExactRefCountedCastTest, RefSubclassThrows)
{
    auto object = New<TMoreDerived>();
    TBase& base = *object;
    EXPECT_THROW((void)ExactRefCountedCast<TDerived>(base), std::bad_cast);
    EXPECT_EQ(&ExactRefCountedCast<TMoreDerived>(base), object.Get());
}

TEST(TExactRefCountedCastTest, RefConstnessPreserved)
{
    auto object = New<TDerived>();
    const TBase& base = *object;
    static_assert(std::is_same_v<decltype(ExactRefCountedCast<TDerived>(base)), const TDerived&>);
    EXPECT_EQ(&ExactRefCountedCast<TDerived>(base), object.Get());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
