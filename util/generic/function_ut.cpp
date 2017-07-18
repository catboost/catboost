#include "function.h"
#include "typetraits.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TestFunctionSignature) {
    int FF(double x) {
        return (int)x;
    }

    int FFF(double x, char xx) {
        return (int)x + (int)xx;
    }

    struct A {
        int F(double x) {
            return FF(x);
        }
    };

    SIMPLE_UNIT_TEST(TestPlainFunc) {
        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(FF)>, decltype(FF));
    }

    SIMPLE_UNIT_TEST(TestMethod) {
        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(&A::F)>, decltype(FF));
    }

    SIMPLE_UNIT_TEST(TestLambda) {
        auto f = [](double x) -> int {
            return FF(x);
        };

        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(f)>, decltype(FF));
    }

    SIMPLE_UNIT_TEST(TestFunction) {
        std::function<int(double)> f(FF);

        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(f)>, decltype(FF));
    }

    template <class F>
    void TestCT() {
#define FA(x) TFunctionArg<F, x>

        UNIT_ASSERT_TYPES_EQUAL(FA(0), double);
        UNIT_ASSERT_TYPES_EQUAL(FA(1), char);
        UNIT_ASSERT_TYPES_EQUAL(TFunctionResult<F>, int);

#undef FA
    }

    SIMPLE_UNIT_TEST(TestTypeErasureTraits) {
        TestCT<std::function<int(double, char)>>();
    }

    SIMPLE_UNIT_TEST(TestPlainFunctionTraits) {
        TestCT<decltype(FFF)>();
    }

    SIMPLE_UNIT_TEST(TestLambdaTraits) {
        auto fff = [](double xx, char xxx) -> int {
            return FFF(xx, xxx);
        };

        TestCT<decltype(fff)>();
    }
}
