#include "utility.h"
#include "ymath.h"

#include <library/unittest/registar.h>

class TUtilityTest: public TTestBase {
    UNIT_TEST_SUITE(TUtilityTest);
    UNIT_TEST(TestSwapPrimitive)
    UNIT_TEST(TestSwapClass)
    UNIT_TEST(TestMaxMin)
    UNIT_TEST(TestMean)
    UNIT_TEST_SUITE_END();

    class TTest {
    public:
        inline TTest(int val)
            : Val(val)
        {
        }

        inline void Swap(TTest& t) {
            DoSwap(Val, t.Val);
        }

        int Val;

    private:
        TTest(const TTest&);
        TTest& operator=(const TTest&);
    };

private:
    inline void TestSwapPrimitive() {
        int i = 0;
        int j = 1;

        DoSwap(i, j);

        UNIT_ASSERT_EQUAL(i, 1);
        UNIT_ASSERT_EQUAL(j, 0);
    }

    inline void TestSwapClass() {
        TTest i(0);
        TTest j(1);

        DoSwap(i, j);

        UNIT_ASSERT_EQUAL(i.Val, 1);
        UNIT_ASSERT_EQUAL(j.Val, 0);
    }

    inline void TestMaxMin() {
        static_assert(Min(10, 3, 8) == 3, "Min doesn't work");
        static_assert(Max(10, 3, 8) == 10, "Max doesn't work");
        UNIT_ASSERT_EQUAL(Min(10, 3, 8), 3);
        UNIT_ASSERT_EQUAL(Max(3.5, 4.2, 8.1, 99.025, 0.33, 29.0), 99.025);
    }

    inline void TestMean() {
        UNIT_ASSERT_EQUAL(Mean(5), 5);
        UNIT_ASSERT_EQUAL(Mean(1, 2, 3), 2);
        UNIT_ASSERT_EQUAL(Mean(6, 5, 4), 5);
        UNIT_ASSERT_EQUAL(Mean(1, 2), 1.5);
        UNIT_ASSERT(Abs(Mean(1., 2., 7.5) - 3.5) < std::numeric_limits<double>::epsilon());
    }
};

UNIT_TEST_SUITE_REGISTRATION(TUtilityTest);
