#include "l1_distance.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>
#include <util/random/fast.h>

Y_UNIT_TEST_SUITE(TL1DistanceTestSuite) {
    inline void RandomNumber(TReallyFastRng32 & rng, i8 & value) {
        value = rng.Uniform(~((ui8)0));
    };

    inline void RandomNumber(TReallyFastRng32 & rng, ui8 & value) {
        value = rng.Uniform(~((ui8)0));
    };

    inline void RandomNumber(TReallyFastRng32 & rng, i32 & value) {
        value = rng.Uniform(~((ui32)0));
    };

    inline void RandomNumber(TReallyFastRng32 & rng, ui32 & value) {
        value = rng.Uniform(~((ui32)0));
    };

    inline void RandomNumber(TReallyFastRng32 & rng, float& value) {
        value = rng.GenRandReal1();
    };

    inline void RandomNumber(TReallyFastRng32 & rng, double& value) {
        value = rng.GenRandReal1();
    };

    template <typename Number>
    void FillWithRandomNumbers(Number * dst, int seed, int length) {
        TReallyFastRng32 Rnd(seed);
        for (int i = 0; i < length; ++i) {
            RandomNumber(Rnd, dst[i]);
        }
    }

    template <typename Res, typename IRes, typename Int>
    Res SimpleL1Dist(const Int* lhs, const Int* rhs, int length) {
        Res sum = 0;
        for (int i = 0; i < length; ++i) {
            IRes diff = static_cast<IRes>(lhs[i]) - static_cast<IRes>(rhs[i]);
            sum += diff >= 0 ? diff : -diff;
        }
        return sum;
    }

    bool Eq(ui64 a, ui64 b) {
        return a == b;
    }

    bool Eq(ui32 a, ui32 b) {
        return a == b;
    }

    bool Eq(double a, double b) {
        return std::fabs(a - b) < 0.00001;
    }

    bool Eq(float a, float b) {
        return (std::fabs(a - b) < 0.0001);
    }

    template <typename Res, typename IRes, typename Number, size_t seed>
    bool Test() {
        TVector<Number> a(100);
        TVector<Number> b(100);
        FillWithRandomNumbers(a.data(), seed, 100);
        FillWithRandomNumbers(a.data(), ~seed & 0xffff, 100);

        for (ui32 i = 0; i < 30; i++) {
            for (ui32 length = 1; length + i + 1 < a.size(); ++length) {
                if (!Eq(L1Distance(a.data() + i, b.data() + i, length), SimpleL1Dist<Res, IRes, Number>(a.data() + i, b.data() + i, length)))
                    return false;
                if (!Eq(L1DistanceSlow(a.data() + i, b.data() + i, length), SimpleL1Dist<Res, IRes, Number>(a.data() + i, b.data() + i, length)))
                    return false;
            }
        }

        return true;
    }

    Y_UNIT_TEST(TestL1Dist_i8) {
        UNIT_ASSERT((Test<ui32, i32, i8, 117>()));
    }

    Y_UNIT_TEST(TestL1Dist_ui8) {
        UNIT_ASSERT((Test<ui32, i32, ui8, 117>()));
    }

    Y_UNIT_TEST(TestL1Dist_i32) {
        UNIT_ASSERT((Test<ui64, i64, i32, 13>()));
    }

    Y_UNIT_TEST(TestL1Dist_ui32) {
        UNIT_ASSERT((Test<ui64, i64, ui32, 13>()));
    }

    Y_UNIT_TEST(TestL1Dist_float) {
        UNIT_ASSERT((Test<float, float, float, 19>()));
    }

    Y_UNIT_TEST(TestL1Dist_double) {
        UNIT_ASSERT((Test<double, double, double, 753>()));
    }

    Y_UNIT_TEST(TestL1Dist_zero_length) {
        UNIT_ASSERT(L1Distance(static_cast<const i8*>(nullptr), nullptr, 0) == 0);
        UNIT_ASSERT(L1Distance(static_cast<const ui8*>(nullptr), nullptr, 0) == 0);
        UNIT_ASSERT(L1DistanceUI4(static_cast<const ui8*>(nullptr), nullptr, 0) == 0);
        UNIT_ASSERT(L1Distance(static_cast<const i32*>(nullptr), nullptr, 0) == 0);
        UNIT_ASSERT(L1Distance(static_cast<const ui32*>(nullptr), nullptr, 0) == 0);
        UNIT_ASSERT(Eq(L1Distance(static_cast<const float*>(nullptr), nullptr, 0), static_cast<float>(0.0)));
        UNIT_ASSERT(Eq(L1Distance(static_cast<const double*>(nullptr), nullptr, 0), 0.0));

        UNIT_ASSERT(L1DistanceSlow(static_cast<const i8*>(nullptr), nullptr, 0) == 0);
        UNIT_ASSERT(L1DistanceSlow(static_cast<const ui8*>(nullptr), nullptr, 0) == 0);
        UNIT_ASSERT(L1DistanceUI4Slow(static_cast<const ui8*>(nullptr), nullptr, 0) == 0);
        UNIT_ASSERT(L1DistanceSlow(static_cast<const i32*>(nullptr), nullptr, 0) == 0);
        UNIT_ASSERT(L1DistanceSlow(static_cast<const ui32*>(nullptr), nullptr, 0) == 0);
        UNIT_ASSERT(Eq(L1DistanceSlow(static_cast<const float*>(nullptr), nullptr, 0), static_cast<float>(0.0)));
        UNIT_ASSERT(Eq(L1DistanceSlow(static_cast<const double*>(nullptr), nullptr, 0), 0.0));
    }

    template <typename Number>
    bool Test1() {
        Number n1;
        Number n2 = 0;
        FillWithRandomNumbers(&n1, 666, 1);

        return Eq(L1Distance(&n1, &n2, 1), (n1 < 0 ? -n1 : n1)) && Eq(L1DistanceSlow(&n1, &n2, 1), (n1 < 0 ? -n1 : n1));
    }

    template <>
    bool Test1<ui32>() {
        ui32 n1;
        ui32 n2 = 0;
        FillWithRandomNumbers(&n1, 666, 1);

        return Eq(L1Distance(&n1, &n2, 1), (ui64)n1) && Eq(L1DistanceSlow(&n1, &n2, 1), (ui64)n1);
    }

    Y_UNIT_TEST(TestL1Dist_length1) {
        UNIT_ASSERT(Test1<i8>());
        UNIT_ASSERT(Test1<ui8>());
        UNIT_ASSERT(Test1<i32>());
        UNIT_ASSERT(Test1<ui32>());
        UNIT_ASSERT(Test1<float>());
        UNIT_ASSERT(Test1<double>());
    }

    Y_UNIT_TEST(TestL1DistUI4_length1) {
        ui8 n1;
        ui8 n2 = 0;
        FillWithRandomNumbers(&n1, 888, 1);
        UNIT_ASSERT_VALUES_EQUAL(L1DistanceUI4(&n1, &n2, 1), (n1 & 0x0f) + ((n1 & 0xf0) >> 4));
        UNIT_ASSERT_VALUES_EQUAL(L1DistanceUI4Slow(&n1, &n2, 1), (n1 & 0x0f) + ((n1 & 0xf0) >> 4));
    }

    Y_UNIT_TEST(TestL1DistUI4_length96) {
        TVector<ui8> n1(96);
        TVector<ui8> n2(96);
        FillWithRandomNumbers(n1.data(), 666, 96);
        FillWithRandomNumbers(n2.data(), 222, 96);

        UNIT_ASSERT_VALUES_EQUAL(L1DistanceUI4(n1.data(), n2.data(), 96), L1DistanceUI4Slow(n1.data(), n2.data(), 96));
    }

    Y_UNIT_TEST(TestL1DistUI4_length72) {
        TVector<ui8> n1(72);
        TVector<ui8> n2(72);
        FillWithRandomNumbers(n1.data(), 666, 72);
        FillWithRandomNumbers(n2.data(), 222, 72);

        UNIT_ASSERT_VALUES_EQUAL(L1DistanceUI4(n1.data(), n2.data(), 72), L1DistanceUI4Slow(n1.data(), n2.data(), 72));
    }

    Y_UNIT_TEST(TestL1Dist_manual_i8) {
        static i8 a[4] = {0, -128, 100, 127};
        static i8 b[4] = {0, 127, -100, -128};
        UNIT_ASSERT_VALUES_EQUAL(L1Distance(a, b, 4), 0 + 255 + 200 + 255);
    }

    Y_UNIT_TEST(TestL1Dist_manual_ui8) {
        static ui8 a[4] = {0, 255, 0, 100};
        static ui8 b[4] = {0, 0, 255, 140};
        UNIT_ASSERT_VALUES_EQUAL(L1Distance(a, b, 4), 0 + 255 + 255 + 40);
    }

    Y_UNIT_TEST(TestL1Dist_manual_ui4) {
        static ui8 a[4] = {0 + (0 << 4), 15 + (8 << 4), 0 + (5 << 4), 3 + (1 << 4)};
        static ui8 b[4] = {0 + (0 << 4), 0 + (8 << 4), 7 + (0 << 4), 1 + (4 << 4)};
        UNIT_ASSERT_VALUES_EQUAL(L1DistanceUI4(a, b, 4), 0 + 15 + 0 + 7 + 5 + 2 + 3);
    }

    Y_UNIT_TEST(TestL1Dist_manual_i32) {
        static i32 a[4] = {0, -128, 100, 127};
        static i32 b[4] = {0, 127, -100, -128};
        UNIT_ASSERT_VALUES_EQUAL(L1Distance(a, b, 4), 0 + 255 + 200 + 255);
    }

    Y_UNIT_TEST(TestL1Dist_manual_ui32) {
        static ui32 a[4] = {0, 0xffffffff, 0, 10000000};
        static ui32 b[4] = {0, 0, 0xffffffff, 14000000};
        UNIT_ASSERT_VALUES_EQUAL(L1Distance(a, b, 4), 0 + 0xfffffffful + 0xfffffffful + 4000000);
    }
}
