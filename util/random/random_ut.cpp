#include "random.h"

#include <library/unittest/registar.h>

#include <util/generic/ylimits.h>

template <class T>
static inline void AssertRange(T v, T r1, T r2) {
    UNIT_ASSERT(v >= r1);
    UNIT_ASSERT(v < r2);
}

SIMPLE_UNIT_TEST_SUITE(TRandomNumberTest) {
    template <typename T>
    void TestAll(T n) {
        for (T i = 0; i < n; ++i) {
            while (RandomNumber<T>(n) != i) {
            }
        }
    }

    template <typename T>
    void TestSome(T n) {
        for (int i = 0; i < 100; ++i) {
            UNIT_ASSERT(RandomNumber<T>(n) < n);
        }
    }

    template <typename T>
    void TestType() {
        TestAll<T>(1);
        TestAll<T>(2);
        TestAll<T>(3);
        TestAll<T>(4);
        TestAll<T>(5);
        TestAll<T>(6);
        TestAll<T>(9);
        TestAll<T>(15);
        TestAll<T>(16);
        TestSome<T>(Max<T>());
        TestSome<T>(Max<T>() - 1);
        TestSome<T>(Max<T>() - 2);
        TestSome<T>(Max<T>() - 3);
        TestSome<T>(Max<T>() - 4);
        TestSome<T>(Max<T>() - 5);
        TestSome<T>(Max<T>() - 7);
        TestSome<T>(Max<T>() - 8);
        TestSome<T>(Max<T>() - 2222);
        TestSome<T>(Max<T>() - 22222);
    }

    SIMPLE_UNIT_TEST(TestWithLimit) {
        TestType<unsigned short>();
        TestType<unsigned int>();
        TestType<unsigned long>();
        TestType<unsigned long long>();
    }

    SIMPLE_UNIT_TEST(TestRandomNumberFloat) {
        for (size_t i = 0; i < 1000; ++i) {
            AssertRange<float>(RandomNumber<float>(), 0.0, 1.0);
        }
    }

    SIMPLE_UNIT_TEST(TestRandomNumberDouble) {
        for (size_t i = 0; i < 1000; ++i) {
            AssertRange<double>(RandomNumber<double>(), 0.0, 1.0);
        }
    }

    SIMPLE_UNIT_TEST(TestRandomNumberLongDouble) {
        for (size_t i = 0; i < 1000; ++i) {
            AssertRange<long double>(RandomNumber<long double>(), 0.0, 1.0);
        }
    }

    SIMPLE_UNIT_TEST(TestBoolean) {
        while (RandomNumber<bool>()) {
        }
        while (!RandomNumber<bool>()) {
        }
    }
}
