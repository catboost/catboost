#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/immortal.h>

#include <string>
#include <type_traits>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

static_assert(!std::is_copy_constructible_v<TImmortal<int>>);
static_assert(!std::is_copy_assignable_v<TImmortal<int>>);
static_assert(!std::is_move_constructible_v<TImmortal<int>>);
static_assert(!std::is_move_assignable_v<TImmortal<int>>);

////////////////////////////////////////////////////////////////////////////////

struct TNonTrivial
{
    constexpr explicit TNonTrivial(int value)
        : Value(value)
    { }

    ~TNonTrivial()
    { }

    int Value;
};

constinit TImmortal<TNonTrivial> GlobalNonTrivial(42);

struct TSelfReferential
{
    int Value = 7;
    const int* Pointer = &Value;

    ~TSelfReferential()
    { }
};

constinit TImmortal<TSelfReferential> GlobalSelfReferential;
constinit const int& GlobalSelfReferentialValue = GlobalSelfReferential->Value;

TEST(TImmortalTest, ConstInit)
{
    EXPECT_EQ(GlobalNonTrivial->Value, 42);
    EXPECT_EQ(GlobalSelfReferential->Pointer, &GlobalSelfReferential->Value);
    EXPECT_EQ(&GlobalSelfReferentialValue, &GlobalSelfReferential->Value);
}

TEST(TImmortalTest, NoDestructorCall)
{
    ::testing::TProbeState state;
    {
        TImmortal<::testing::TProbe> probe(&state);
        EXPECT_EQ(probe->State, &state);
    }
    EXPECT_EQ(state.Constructors, 1);
    EXPECT_EQ(state.Destructors, 0);
}

TEST(TImmortalTest, ForwardsArguments)
{
    TImmortal<std::string> immortal(3, 'x');
    EXPECT_EQ(*immortal, "xxx");
    // NB: Short enough for SSO, so nothing leaks.
}

TEST(TImmortalTest, Accessors)
{
    TImmortal<TNonTrivial> immortal(1);
    EXPECT_EQ(immortal.Get(), &*immortal);
    EXPECT_EQ(immortal.Get(), immortal.operator->());

    immortal->Value = 2;
    EXPECT_EQ((*immortal).Value, 2);

    const auto& constImmortal = immortal;
    EXPECT_EQ(constImmortal.Get(), immortal.Get());
    EXPECT_EQ(constImmortal->Value, 2);
    EXPECT_EQ((*constImmortal).Value, 2);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
