#include <library/overloaded/overloaded.h>

#include <library/cpp/unittest/registar.h>

#include <util/generic/variant.h>
#include <util/generic/algorithm.h>

#include <tuple>

namespace {
    struct TType1 {};
    struct TType2 {};
    struct TType3 {};
}

Y_UNIT_TEST_SUITE(TOverloadedTest) {
    Y_UNIT_TEST(StaticTest) {
        auto f = TOverloaded{
            [](const TType1&) {},
            [](const TType2&) {},
            [](const TType3&) {}
        };
        using F = decltype(f);
        static_assert(std::is_invocable_v<F, TType1>);
        static_assert(std::is_invocable_v<F, TType2>);
        static_assert(std::is_invocable_v<F, TType3>);
        static_assert(!std::is_invocable_v<F, int>);
        static_assert(!std::is_invocable_v<F, double>);
    }

    Y_UNIT_TEST(VariantTest) {
        TVariant<int, double, TType1> v = 5;
        int res = 0;
        Visit(TOverloaded{
            [&](int val) { res = val; },
            [&](double) { res = -1; },
            [&](TType1) { res = -1; }
        }, v);
        UNIT_ASSERT_EQUAL(res, 5);
    }

    Y_UNIT_TEST(TupleTest) {
        std::tuple<int, double, bool, int> t{ 5, 3.14, true, 20 };
        TString res;

        ForEach(t, TOverloaded{
            [&](int val) { res += "(int) " + ToString(val) + ' '; },
            [&](double val) { res += "(double) " + ToString(val) + ' '; },
            [&](bool val) { res += "(bool) " + ToString(val) + ' '; },
        });

        UNIT_ASSERT_EQUAL(res, "(int) 5 (double) 3.14 (bool) 1 (int) 20 ");
    }
}
