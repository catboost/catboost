#include <catboost/private/libs/functools/forward_as_const.h>

#include <library/cpp/testing/unittest/gtest.h>

enum class ETestEnum {
    OptOne,
    OptTwo,
    UnusedOption
};

using TTestIntOption = TIntOption<int, 1, 3, 4>;
using TTestEnumOption = TIntOption<ETestEnum, ETestEnum::OptOne, ETestEnum::OptTwo>;

Y_UNIT_TEST_SUITE(TestTIntOption) {
    Y_UNIT_TEST(GoodInit) {
        for (int i : {1, 3, 4}) {
            UNIT_ASSERT(TTestIntOption(i).Value == i);
            UNIT_ASSERT(TTestIntOption::CheckValue(i));
        }
        for (auto option : {ETestEnum::OptOne, ETestEnum::OptTwo}) {
            UNIT_ASSERT(TTestEnumOption(option).Value == option);
            UNIT_ASSERT(TTestEnumOption::CheckValue(option));
        }
    }


    Y_UNIT_TEST(BadInit) {
        using TTestIntOption = TIntOption<int, 1, 2, 3>;
        for (int i : {0, 4, 100}) {
            UNIT_ASSERT_EXCEPTION(TTestIntOption(i), yexception);
            UNIT_ASSERT(!TTestIntOption::CheckValue(i));
        }
        UNIT_ASSERT_EXCEPTION(TTestEnumOption(ETestEnum::UnusedOption), yexception);
        UNIT_ASSERT(!TTestEnumOption::CheckValue(ETestEnum::UnusedOption));

        UNIT_ASSERT_EXCEPTION(TTestEnumOption((ETestEnum)100), yexception);
        UNIT_ASSERT(!TTestEnumOption::CheckValue((ETestEnum)100));
    }
}

Y_UNIT_TEST_SUITE(TestForwarding) {
    Y_UNIT_TEST(CheckOptionsSum) {
        auto sumConstants = [](auto boolConst, auto intConst, auto enumConst) -> int {
            constexpr int sum = ((boolConst ? 10 : 0) + 2 * intConst + (int)enumConst.value);
            return sum;
        };
        UNIT_ASSERT_EQUAL(
                ForwardArgsAsIntegralConst(
                        sumConstants,
                        true,
                        TTestIntOption(4),
                        TTestEnumOption(ETestEnum::OptTwo)
                ),
                19
        );
        UNIT_ASSERT_EQUAL(
                ForwardArgsAsIntegralConst(
                        sumConstants,
                        false,
                        TTestIntOption(1),
                        TTestEnumOption(ETestEnum::OptOne)
                ),
                2
        );
    }
}
