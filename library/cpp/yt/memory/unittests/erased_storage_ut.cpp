#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/erased_storage.h>

#include <library/cpp/int128/int128.h>

#include <library/cpp/yt/misc/guid.h>

#include <util/generic/string.h>
#include <util/system/types.h>

#include <vector>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

struct TWithFieldInitalizer
{
    // NB: This class is not trivially default constructible.
    int Field{};
};

static_assert(!std::is_trivially_default_constructible_v<TWithFieldInitalizer>);

struct TCopyWithSideEffects
{
    TCopyWithSideEffects(const TCopyWithSideEffects&)
    { }
};

static_assert(!std::is_trivially_copy_constructible_v<TCopyWithSideEffects>);

struct TWithSubStruct
{
    TWithFieldInitalizer Field;
};

class TWithPrivateMembers
{
public:
    TWithPrivateMembers() = default;

private:
    [[maybe_unused]] std::array<std::byte, 8> Data_;
};

// Overshadow to bind template parameter.
inline constexpr size_t TestSize = 32;

template <class T>
concept CTriviallyErasable = ::NYT::CTriviallyErasable<T, TestSize>;
using TErasedStorage = ::NYT::TErasedStorage<TestSize>;

////////////////////////////////////////////////////////////////////////////////

TEST(TErasedStorageTest, Types)
{
    static_assert(CTriviallyErasable<int>);
    static_assert(CTriviallyErasable<i32>);
    static_assert(CTriviallyErasable<i64>);
    static_assert(CTriviallyErasable<i128>);
    static_assert(CTriviallyErasable<std::array<i128, 2>>);
    static_assert(CTriviallyErasable<TGuid>);
    static_assert(CTriviallyErasable<void*>);
    static_assert(CTriviallyErasable<double>);
    static_assert(CTriviallyErasable<const char*>);
    static_assert(CTriviallyErasable<TWithFieldInitalizer>);
    static_assert(CTriviallyErasable<TWithSubStruct>);
    static_assert(CTriviallyErasable<TWithPrivateMembers>);
    static_assert(CTriviallyErasable<char[8]>);

    static_assert(!CTriviallyErasable<TString>);
    static_assert(!CTriviallyErasable<std::vector<int>>);
    static_assert(!CTriviallyErasable<std::array<i128, 3>>);
    static_assert(!CTriviallyErasable<int&>);
    static_assert(!CTriviallyErasable<const int&>);
    static_assert(!CTriviallyErasable<int&&>);
    static_assert(!CTriviallyErasable<TCopyWithSideEffects>);
}

TEST(TErasedStorageTest, JustWorks)
{
    int var = 42;

    TErasedStorage stor(var);
    EXPECT_EQ(stor.AsConcrete<int>(), 42);

    var = 66;
    EXPECT_EQ(stor.AsConcrete<int>(), 42);
}

TEST(TErasedStorageTest, CopyAssign)
{
    int var = 42;
    TErasedStorage stor(var);
    EXPECT_EQ(stor.AsConcrete<int>(), 42);

    {
        int anotherVar = 77;
        stor = TErasedStorage(anotherVar);
    }
    EXPECT_EQ(stor.AsConcrete<int>(), 77);

    double thirdVar = 9.92145214;
    stor = TErasedStorage(thirdVar);
    EXPECT_DOUBLE_EQ(stor.AsConcrete<double>(), 9.92145214);
}

TEST(TErasedStorageTest, Pointer)
{
    TString message("Hello world");
    TErasedStorage stor(&message);

    EXPECT_EQ(*stor.AsConcrete<TString*>(), TString("Hello world"));
    message = "Goodbye world";

    EXPECT_EQ(*stor.AsConcrete<TString*>(), "Goodbye world");
}

TEST(TErasedStorageTest, MutateStorage)
{
    int var = 0;
    TErasedStorage stor(var);
    EXPECT_EQ(stor.AsConcrete<int>(), 0);

    auto& ref = stor.AsConcrete<int>();
    ref = 88;

    EXPECT_EQ(stor.AsConcrete<int>(), 88);
}

TEST(TErasedStorageTest, EqualityComparison)
{
    struct TWidget
    {
        alignas(8) int Value;

        alignas(16) bool Flag;
    } widget{1, false};

    TErasedStorage stor1(widget);
    TErasedStorage stor2(widget);

    EXPECT_EQ(stor1, stor2);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
