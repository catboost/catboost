#include "random.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/ylimits.h>

template <class T>
static inline void AssertRange(T v, T r1, T r2) {
    UNIT_ASSERT(v >= r1);
    UNIT_ASSERT(v < r2);
}

Y_UNIT_TEST_SUITE(TRandomNumberTest) {
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

    Y_UNIT_TEST(TestWithLimit) {
        TestType<unsigned short>();
        TestType<unsigned int>();
        TestType<unsigned long>();
        TestType<unsigned long long>();
    }

    Y_UNIT_TEST(TestRandomNumberFloat) {
        for (size_t i = 0; i < 1000; ++i) {
            AssertRange<float>(RandomNumber<float>(), 0.0, 1.0);
        }
    }

    Y_UNIT_TEST(TestRandomNumberDouble) {
        for (size_t i = 0; i < 1000; ++i) {
            AssertRange<double>(RandomNumber<double>(), 0.0, 1.0);
        }
    }

    Y_UNIT_TEST(TestRandomNumberLongDouble) {
        for (size_t i = 0; i < 1000; ++i) {
            AssertRange<long double>(RandomNumber<long double>(), 0.0, 1.0);
        }
    }

    Y_UNIT_TEST(TestBoolean) {
        while (RandomNumber<bool>()) {
        }
        while (!RandomNumber<bool>()) {
        }
    }

    Y_UNIT_TEST(TestResetSeed) {
        SetRandomSeed(42);
        for (const ui32 el : {
                 102,
                 179,
                 92,
                 14,
                 106,
                 71,
                 188,
                 20,
                 102,
                 121,
                 210,
                 214,
                 74,
                 202,
                 87,
                 116,
                 99,
                 103,
                 151,
                 130,
                 149,
                 52,
                 1,
                 87,
                 235,
                 157,
                 37,
                 129,
                 191,
                 187,
                 20,
                 160,
                 203,
                 57,
                 21,
                 252,
                 235,
                 88,
                 48,
                 218,
                 58,
                 254,
                 169,
                 255,
                 219,
                 187,
                 207,
                 14,
                 189,
                 189,
                 174,
                 189,
                 50,
                 107,
                 54,
                 243,
                 63,
                 248,
                 130,
                 228,
                 50,
                 134,
                 20,
                 72,
             }) {
            UNIT_ASSERT_EQUAL(RandomNumber<ui32>(1 << 8), el);
        }
    }
} // Y_UNIT_TEST_SUITE(TRandomNumberTest)
