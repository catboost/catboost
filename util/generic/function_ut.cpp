#include "function.h"
#include "typetraits.h"

#include <library/cpp/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestFunctionSignature) {
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

    Y_UNIT_TEST(TestPlainFunc) {
        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(FF)>, decltype(FF));
    }

    Y_UNIT_TEST(TestMethod) {
        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(&A::F)>, decltype(FF));
    }

    Y_UNIT_TEST(TestLambda) {
        auto f = [](double x) -> int {
            return FF(x);
        };

        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(f)>, decltype(FF));
    }

    Y_UNIT_TEST(TestFunction) {
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

    Y_UNIT_TEST(TestTypeErasureTraits) {
        TestCT<std::function<int(double, char)>>();
    }

    Y_UNIT_TEST(TestPlainFunctionTraits) {
        TestCT<decltype(FFF)>();
    }

    Y_UNIT_TEST(TestLambdaTraits) {
        auto fff = [](double xx, char xxx) -> int {
            return FFF(xx, xxx);
        };

        TestCT<decltype(fff)>();
    }
}
