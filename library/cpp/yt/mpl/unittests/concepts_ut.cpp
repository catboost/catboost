#include <library/cpp/yt/mpl/concepts.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace NYT::NMpl {
namespace {

////////////////////////////////////////////////////////////////////////////////

struct TScalableValue
{
    TScalableValue operator*(double scalar) const;
};

template <class T>
struct TBase
{ };

struct TDerived
    : public TBase<int>
{ };

struct TUnrelated
{ };

////////////////////////////////////////////////////////////////////////////////

TEST(TConceptsTest, Scalable)
{
    static_assert(CScalable<TScalableValue, double>);
    static_assert(CScalable<double, double>);
    static_assert(!CScalable<TUnrelated, double>);
}

TEST(TConceptsTest, Invocable)
{
    auto returnsInt = [] (int value) { return value; };
    auto returnsVoid = [] { };
    auto noexceptVoid = [] () noexcept { };

    static_assert(CInvocable<decltype(returnsInt), int(int)>);
    static_assert(!CInvocable<decltype(returnsInt), void(int)>);
    static_assert(!CInvocable<decltype(returnsInt), int(std::string)>);

    static_assert(CInvocable<decltype(returnsVoid), void()>);
    static_assert(!CInvocable<decltype(returnsVoid), void() noexcept>);
    static_assert(CInvocable<decltype(noexceptVoid), void() noexcept>);
}

TEST(TConceptsTest, OneOf)
{
    static_assert(COneOf<int, int, double>);
    static_assert(COneOf<int, double, int>);
    static_assert(!COneOf<int, double, std::string>);
    static_assert(!COneOf<int>);
}

TEST(TConceptsTest, Distinct)
{
    static_assert(CDistinct<>);
    static_assert(CDistinct<int>);
    static_assert(CDistinct<int, double, std::string>);
    static_assert(!CDistinct<int, double, int>);
}

TEST(TConceptsTest, DerivedFromSpecializationOf)
{
    static_assert(CDerivedFromSpecializationOf<TDerived, TBase>);
    static_assert(CDerivedFromSpecializationOf<TBase<int>, TBase>);
    static_assert(!CDerivedFromSpecializationOf<TUnrelated, TBase>);
}

TEST(TConceptsTest, StdVector)
{
    static_assert(CStdVector<std::vector<int>>);
    static_assert(CStdVector<std::vector<std::string>>);
    static_assert(!CStdVector<std::string>);
    static_assert(!CStdVector<int>);
}

TEST(TConceptsTest, AssociativeAndMapping)
{
    static_assert(CAssociative<std::set<int>>);
    static_assert(CAssociative<std::map<int, int>>);
    static_assert(!CAssociative<std::vector<int>>);

    static_assert(CMapping<std::map<int, int>>);
    static_assert(CMapping<std::unordered_map<int, int>>);
    static_assert(!CMapping<std::set<int>>);
    static_assert(!CMapping<std::vector<int>>);
}

TEST(TConceptsTest, Constness)
{
    static_assert(CConst<const int>);
    static_assert(!CConst<int>);

    static_assert(CNonConst<int>);
    static_assert(!CNonConst<const int>);
}

TEST(TConceptsTest, RawPtr)
{
    static_assert(CRawPtr<int*>);
    static_assert(CRawPtr<const int*>);
    static_assert(!CRawPtr<int>);

    static_assert(CConstRawPtr<const int*>);
    static_assert(!CConstRawPtr<int*>);

    static_assert(CMutableRawPtr<int*>);
    static_assert(!CMutableRawPtr<const int*>);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT::NMpl
