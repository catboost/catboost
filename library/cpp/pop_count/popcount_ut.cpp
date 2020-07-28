#include "popcount.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/random/random.h>

Y_UNIT_TEST_SUITE(TestPopCount) {
    template <class T>
    static inline ui32 SlowPopCount(T t) {
        ui32 ret = 0;

        while (t) {
            if (t & T(1)) {
                ++ret;
            }

            t = t >> 1;
        }

        return ret;
    }

    template <class T>
    static inline void Test() {
        for (size_t i = 0; i < 10000; ++i) {
            const T rndv = RandomNumber<T>();

            UNIT_ASSERT_VALUES_EQUAL(SlowPopCount(rndv), PopCount(rndv));
        }
    }

    Y_UNIT_TEST(Test8) {
        Test<ui8>();
    }

    Y_UNIT_TEST(Test16) {
        Test<ui16>();
    }

    Y_UNIT_TEST(Test32) {
        Test<ui32>();
    }

    Y_UNIT_TEST(Test64) {
        Test<ui64>();
    }

    Y_UNIT_TEST(TestPopCount) {
        UNIT_ASSERT_VALUES_EQUAL(PopCount(0), 0);
        UNIT_ASSERT_VALUES_EQUAL(PopCount(1), 1);
        UNIT_ASSERT_VALUES_EQUAL(PopCount(1 << 10), 1);
        UNIT_ASSERT_VALUES_EQUAL(PopCount((1 << 10) + 1), 2);
        UNIT_ASSERT_VALUES_EQUAL(PopCount(0xFFFF), 16);
        UNIT_ASSERT_VALUES_EQUAL(PopCount(0xFFFFFFFF), 32);
        UNIT_ASSERT_VALUES_EQUAL(PopCount(0x55555555), 16);

        UNIT_ASSERT_VALUES_EQUAL(0, PopCount(0ULL));
        UNIT_ASSERT_VALUES_EQUAL(1, PopCount(1ULL));
        UNIT_ASSERT_VALUES_EQUAL(16, PopCount(0xAAAAAAAAULL));
        UNIT_ASSERT_VALUES_EQUAL(32, PopCount(0xFFFFFFFFULL));
        UNIT_ASSERT_VALUES_EQUAL(32, PopCount(0xAAAAAAAAAAAAAAAAULL));
        UNIT_ASSERT_VALUES_EQUAL(64, PopCount(0xFFFFFFFFFFFFFFFFULL));

        ui64 v = 0;

        for (int i = 0; i < 64; v |= 1ULL << i, ++i) {
            UNIT_ASSERT_VALUES_EQUAL(i, PopCount(v));
        }
    }
}
