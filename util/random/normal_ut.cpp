#include "normal.h"
#include "fast.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>

#include <functional>

Y_UNIT_TEST_SUITE(TestNormalDistribution) {
    Y_UNIT_TEST(TestDefined) {
        volatile auto x = NormalRandom<float>(0, 1) + NormalRandom<double>(0, 1) + NormalRandom<long double>(0, 1);

        (void)x;
    }

    template <class T>
    static void TestMD(std::function<T()> f, T m, T d) {
        TVector<T> v;

        v.reserve(20000);

        for (size_t i = 0; i < 20000; ++i) {
            v.push_back(f());
        }

        long double mm = 0;
        long double vv = 0;

        for (auto x : v) {
            mm += x;
        }

        mm /= v.size();

        for (auto x : v) {
            vv += (mm - x) * (mm - x);
        }

        vv /= v.size();

        long double dd = std::sqrt(vv);

        UNIT_ASSERT_DOUBLES_EQUAL(m, mm, (m + 1) * 0.05);
        UNIT_ASSERT_DOUBLES_EQUAL(d, dd, (d + 1) * 0.05);
    }

    Y_UNIT_TEST(Test1) {
        TestMD<float>(&StdNormalRandom<float>, 0, 1);
        TestMD<double>(&StdNormalRandom<double>, 0, 1);
        TestMD<long double>(&StdNormalRandom<long double>, 0, 1);
    }

    template <class T>
    std::function<T()> GenFunc1(T m, T d) {
        return [m, d]() {
            return NormalRandom<T>(m, d);
        };
    }

    template <class T>
    std::function<T()> GenFunc2(T m, T d) {
        TFastRng<ui64> rng(17);

        return [rng, m, d]() mutable {
            return NormalDistribution<T>(rng, m, d);
        };
    }

    Y_UNIT_TEST(Test2) {
        TestMD<float>(GenFunc1<float>(2, 3), 2, 3);
        TestMD<double>(GenFunc1<double>(3, 4), 3, 4);
        TestMD<long double>(GenFunc1<long double>(4, 5), 4, 5);
    }

    Y_UNIT_TEST(Test3) {
        TestMD<float>(GenFunc2<float>(20, 30), 20, 30);
        TestMD<double>(GenFunc2<double>(30, 40), 30, 40);
        TestMD<long double>(GenFunc2<long double>(40, 50), 40, 50);
    }
} // Y_UNIT_TEST_SUITE(TestNormalDistribution)
