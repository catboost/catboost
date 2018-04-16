#include "dot_product.h"

#include <library/unittest/registar.h>

#include <util/generic/vector.h>
#include <util/random/fast.h>

#include <cmath>

SIMPLE_UNIT_TEST_SUITE(TDocProductTestSuite) {
    const double EPSILON = 0.00001;
    template <class Num>
    void FillWithRandomNumbers(Num * dst, int seed, int length) {
        TReallyFastRng32 Rnd(seed);
        Num maxNum = ~Num(0);
        for (int i = 0; i < length; ++i) {
            dst[i] = Rnd.Uniform(maxNum);
        }
    }

    template <class Num>
    void FillWithRandomFloats(Num * dst, int seed, int length) {
        TReallyFastRng32 Rnd(seed);
        for (int i = 0; i < length; ++i) {
            dst[i] = Rnd.GenRandReal1();
        }
    }

    template <class Res, class Int>
    Res SimpleDotProduct(const Int* lhs, const Int* rhs, int length) {
        Res sum = 0;
        for (int i = 0; i < length; ++i) {
            sum += static_cast<Res>(lhs[i]) * static_cast<Res>(rhs[i]);
        }
        return sum;
    }

    SIMPLE_UNIT_TEST(TestDotProduct8) {
        TVector<i8> a(100);
        FillWithRandomNumbers(~a, 179, 100);
        TVector<i8> b(100);
        FillWithRandomNumbers(~b, 239, 100);

        for (int i = 0; i < 30; ++i) {
            for (ui32 length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT_EQUAL(DotProduct(~a + i, ~b + i, length), (SimpleDotProduct<i32, i8>(~a + i, ~b + i, length)));
                UNIT_ASSERT_EQUAL(DotProductSlow(~a + i, ~b + i, length), (SimpleDotProduct<i32, i8>(~a + i, ~b + i, length)));
            }
        }
    }

    SIMPLE_UNIT_TEST(TestDotProduct32) {
        TVector<i32> a(100);
        FillWithRandomNumbers(~a, 179, 100);
        TVector<i32> b(100);
        FillWithRandomNumbers(~b, 239, 100);

        for (int i = 0; i < 30; ++i) {
            for (ui32 length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT_EQUAL(DotProduct(~a + i, ~b + i, length), (SimpleDotProduct<i64, i32>(~a + i, ~b + i, length)));
                UNIT_ASSERT_EQUAL(DotProductSlow(~a + i, ~b + i, length), (SimpleDotProduct<i64, i32>(~a + i, ~b + i, length)));
            }
        }
    }

    SIMPLE_UNIT_TEST(TestDotProductf) {
        TVector<float> a(100);
        FillWithRandomFloats(~a, 179, 100);
        TVector<float> b(100);
        FillWithRandomFloats(~b, 239, 100);

        for (int i = 0; i < 30; ++i) {
            for (ui32 length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT(std::fabs(DotProduct(~a + i, ~b + i, length) - (SimpleDotProduct<float, float>(~a + i, ~b + i, length))) < EPSILON);
                UNIT_ASSERT(std::fabs(DotProductSlow(~a + i, ~b + i, length) - (SimpleDotProduct<float, float>(~a + i, ~b + i, length))) < EPSILON);
            }
        }
    }

    SIMPLE_UNIT_TEST(TestL2NormSqaredf) {
        TVector<float> a(100);
        FillWithRandomFloats(~a, 179, 100);
        TVector<float> b(100);
        FillWithRandomFloats(~b, 239, 100);

        for (int i = 0; i < 30; ++i) {
            for (ui32 length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT(std::fabs(L2NormSquared(~a + i, length) - DotProductSlow(~a + i, ~a + i, length)) < EPSILON);
                UNIT_ASSERT(std::fabs(L2NormSquared(~b + i, length) - DotProductSlow(~b + i, ~b + i, length)) < EPSILON);
            }
        }
    }

    SIMPLE_UNIT_TEST(TestDotProductd) {
        TVector<double> a(100);
        FillWithRandomFloats(~a, 179, 100);
        TVector<double> b(100);
        FillWithRandomFloats(~b, 239, 100);

        for (int i = 0; i < 30; ++i) {
            for (ui32 length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT(std::fabs(DotProduct(~a + i, ~b + i, length) - (SimpleDotProduct<double, double>(~a + i, ~b + i, length))) < EPSILON);
                UNIT_ASSERT(std::fabs(DotProductSlow(~a + i, ~b + i, length) - (SimpleDotProduct<double, double>(~a + i, ~b + i, length))) < EPSILON);
            }
        }
    }

    SIMPLE_UNIT_TEST(TestDotProductZeroLength) {
        UNIT_ASSERT_EQUAL(DotProduct(static_cast<const i8*>(nullptr), nullptr, 0), 0);
        UNIT_ASSERT_EQUAL(DotProduct(static_cast<const i32*>(nullptr), nullptr, 0), 0);
        UNIT_ASSERT(std::abs(DotProduct(static_cast<const float*>(nullptr), nullptr, 0)) < EPSILON);
        UNIT_ASSERT(std::abs(DotProduct(static_cast<const double*>(nullptr), nullptr, 0)) < EPSILON);
        UNIT_ASSERT_EQUAL(DotProductSlow(static_cast<const i8*>(nullptr), nullptr, 0), 0);
        UNIT_ASSERT_EQUAL(DotProductSlow(static_cast<const i32*>(nullptr), nullptr, 0), 0);
        UNIT_ASSERT(std::abs(DotProductSlow(static_cast<const float*>(nullptr), nullptr, 0)) < EPSILON);
        UNIT_ASSERT(std::abs(DotProductSlow(static_cast<const double*>(nullptr), nullptr, 0)) < EPSILON);
    }

    SIMPLE_UNIT_TEST(TestDotProductFloatStability) {
        TVector<float> a(1003);
        FillWithRandomFloats(~a, 179, a.size());
        TVector<float> b(1003);
        FillWithRandomFloats(~b, 239, b.size());

        float res = DotProduct(~a, ~b, a.size());

        for (int i = 0; i < 30; ++i)
            UNIT_ASSERT_VALUES_EQUAL(DotProduct(~a, ~b, a.size()), res);

#ifdef _sse_
        UNIT_ASSERT_VALUES_EQUAL(ToString(res), "250.502");
#endif
    }

    SIMPLE_UNIT_TEST(TestDotProductDoubleStability) {
        TVector<double> a(1003);
        FillWithRandomFloats(~a, 13133, a.size());
        TVector<double> b(1003);
        FillWithRandomFloats(~b, 1121, b.size());

        double res = DotProduct(~a, ~b, a.size());

        for (int i = 0; i < 30; ++i)
            UNIT_ASSERT_VALUES_EQUAL(DotProduct(~a, ~b, a.size()), res);

#ifdef _sse_
        UNIT_ASSERT_VALUES_EQUAL(ToString(res), "235.7826026");
#endif
    }

    SIMPLE_UNIT_TEST(TestDotProductCharStability) {
        TVector<i8> a(1003);
        FillWithRandomNumbers(~a, 1079, a.size());
        TVector<i8> b(1003);
        FillWithRandomNumbers(~b, 2139, b.size());

        ui32 res = DotProduct(~a, ~b, a.size());

        for (int i = 0; i < 30; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(DotProduct(~a, ~b, a.size()), res);
            UNIT_ASSERT_VALUES_EQUAL(DotProductSlow(~a, ~b, a.size()), res);
        }

        UNIT_ASSERT_VALUES_EQUAL(res, 90928);
    }
}
