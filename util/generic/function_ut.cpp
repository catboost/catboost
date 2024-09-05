#include "function.h"
#include "typetraits.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestFunctionSignature) {
    int FF(double x) {
        return (int)x;
    }

    int FFN(double x) noexcept {
        return (int)x;
    }

    int FFF(double x, char xx) {
        return (int)x + (int)xx;
    }

    int FFFN(double x, char xx) noexcept {
        return (int)x + (int)xx;
    }

    struct A {
        int F(double x) {
            return FF(x);
        }

        int FN(double x) noexcept {
            return FFN(x);
        }

        int FC(double x) const {
            return FF(x);
        }

        int FCN(double x) const noexcept {
            return FFN(x);
        }

#define Y_FOR_EACH_REF_QUALIFIED_MEMBERS(XX) \
    XX(AsMutLvalue, &, false)                \
    XX(AsMutLvalueN, &, true)                \
    XX(AsMutRvalue, &&, false)               \
    XX(AsMutRvalueN, &&, true)               \
    XX(AsConstLvalue, const&, false)         \
    XX(AsConstLvalueN, const&, true)         \
    XX(AsConstRvalue, const&&, false)        \
    XX(AsConstRvalueN, const&&, true)

#define Y_ADD_MEMBER(name, qualifiers, isNoexcept)       \
    int name(double x) qualifiers noexcept(isNoexcept) { \
        return FF(x);                                    \
    }

        Y_FOR_EACH_REF_QUALIFIED_MEMBERS(Y_ADD_MEMBER)
#undef Y_ADD_MEMBER
    };

    Y_UNIT_TEST(TestPlainFunc) {
        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(FF)>, decltype(FF));
        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(FFN)>, decltype(FF));
    }

    Y_UNIT_TEST(TestMethod) {
        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(&A::F)>, decltype(FF));
        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(&A::FN)>, decltype(FF));
        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(&A::FC)>, decltype(FF));
        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(&A::FCN)>, decltype(FF));

#define Y_CHECK_MEMBER(name, qualifiers, isNoexcept) \
    UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(&A::name)>, decltype(FF));

        Y_FOR_EACH_REF_QUALIFIED_MEMBERS(Y_CHECK_MEMBER)
#undef Y_CHECK_MEMBER
    }

    Y_UNIT_TEST(TestLambda) {
        auto f = [](double x) -> int {
            return FF(x);
        };

        auto fn = [](double x) mutable noexcept -> int {
            return FFN(x);
        };

        auto fcn = [](double x) noexcept -> int {
            return FFN(x);
        };

        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(f)>, decltype(FF));
        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(fn)>, decltype(FF));
        UNIT_ASSERT_TYPES_EQUAL(TFunctionSignature<decltype(fcn)>, decltype(FF));
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
        TestCT<decltype(FFFN)>();
    }

    Y_UNIT_TEST(TestLambdaTraits) {
        auto fff = [](double xx, char xxx) -> int {
            return FFF(xx, xxx);
        };

        auto fffn = [](double xx, char xxx) mutable noexcept -> int {
            return FFFN(xx, xxx);
        };

        auto fffcn = [](double xx, char xxx) noexcept -> int {
            return FFFN(xx, xxx);
        };

        TestCT<decltype(fff)>();
        TestCT<decltype(fffn)>();
        TestCT<decltype(fffcn)>();
    }
} // Y_UNIT_TEST_SUITE(TestFunctionSignature)
