#include <library/cpp/yt/mpl/type_traits.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <string>
#include <vector>

namespace NYT::NMpl {
namespace {

////////////////////////////////////////////////////////////////////////////////

struct TNonPod
{
    std::string String;
};

struct TEmpty
{ };

struct TNonEmpty
{
    int Value;
};

////////////////////////////////////////////////////////////////////////////////

TEST(TTypeTraitsTest, IsPod)
{
    EXPECT_TRUE((TIsPod<char>::value));
    EXPECT_TRUE((TIsPod<int>::value));
    EXPECT_TRUE((TIsPod<short>::value));
    EXPECT_TRUE((TIsPod<long>::value));
    EXPECT_TRUE((TIsPod<float>::value));
    EXPECT_TRUE((TIsPod<double>::value));

    EXPECT_FALSE((TIsPod<TNonPod>::value));
    EXPECT_FALSE((TIsPod<std::string>::value));
}

TEST(TTypeTraitsTest, CallTraits)
{
    static_assert(std::is_same_v<TCallTraits<int>::TType, int>);
    static_assert(std::is_same_v<TCallTraits<double>::TType, double>);
    static_assert(std::is_same_v<TCallTraits<int*>::TType, int*>);

    static_assert(std::is_same_v<TCallTraits<std::string>::TType, const std::string&>);
    static_assert(std::is_same_v<TCallTraits<TNonPod>::TType, const TNonPod&>);
}

TEST(TTypeTraitsTest, IsSpecialization)
{
    static_assert(IsSpecialization<std::vector<int>, std::vector>);
    static_assert(IsSpecialization<std::vector<std::string>, std::vector>);

    static_assert(!IsSpecialization<int, std::vector>);
    static_assert(!IsSpecialization<std::string, std::vector>);
}

TEST(TTypeTraitsTest, IsEmptyClass)
{
    static_assert(IsEmptyClass<TEmpty>());
    static_assert(!IsEmptyClass<TNonEmpty>());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT::NMpl
