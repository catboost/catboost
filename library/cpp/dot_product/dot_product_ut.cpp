#include "dot_product.h"

#include <library/cpp/unittest/registar.h>

#include <library/cpp/sse/sse.h>
#include <util/generic/vector.h>
#include <util/random/fast.h>

#include <cmath>

Y_UNIT_TEST_SUITE(TDocProductTestSuite) {
    const double EPSILON = 0.00001;
    template <class Num>
    void FillWithRandomNumbers(Num * dst, int seed, ui32 length) {
        TReallyFastRng32 Rnd(seed);
        Num maxNum = ~Num(0);
        for (ui32 i = 0; i < length; ++i) {
            dst[i] = Rnd.Uniform(maxNum);
        }
    }

    template <class Num>
    void FillWithRandomFloats(Num * dst, int seed, ui32 length) {
        TReallyFastRng32 Rnd(seed);
        for (ui32 i = 0; i < length; ++i) {
            dst[i] = Rnd.GenRandReal1();
        }
    }

    template <class Res, class Int>
    Res SimpleDotProduct(const Int* lhs, const Int* rhs, ui32 length) {
        Res sum = 0;
        for (ui32 i = 0; i < length; ++i) {
            sum += static_cast<Res>(lhs[i]) * static_cast<Res>(rhs[i]);
        }
        return sum;
    }

    Y_UNIT_TEST(TestDotProduct8) {
        TVector<i8> a(100);
        FillWithRandomNumbers(a.data(), 179, 100);
        TVector<i8> b(100);
        FillWithRandomNumbers(b.data(), 239, 100);

        for (ui32 i = 0; i < 30; ++i) {
            for (ui32 length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT_EQUAL(DotProduct(a.data() + i, b.data() + i, length), (SimpleDotProduct<i32, i8>(a.data() + i, b.data() + i, length)));
                UNIT_ASSERT_EQUAL(DotProductSlow(a.data() + i, b.data() + i, length), (SimpleDotProduct<i32, i8>(a.data() + i, b.data() + i, length)));
            }
        }
    }

    Y_UNIT_TEST(TestDotProduct8u) {
        TVector<ui8> a(100);
        FillWithRandomNumbers(a.data(), 179, 100);
        TVector<ui8> b(100);
        FillWithRandomNumbers(b.data(), 239, 100);

        for (ui32 i = 0; i < 30; ++i) {
            for (ui32 length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT_EQUAL(DotProduct(a.data() + i, b.data() + i, length), (SimpleDotProduct<ui32, ui8>(a.data() + i, b.data() + i, length)));
                UNIT_ASSERT_EQUAL(DotProductSlow(a.data() + i, b.data() + i, length), (SimpleDotProduct<ui32, ui8>(a.data() + i, b.data() + i, length)));
            }
        }
    }

    Y_UNIT_TEST(TestDotProduct32) {
        TVector<i32> a(100);
        FillWithRandomNumbers(a.data(), 179, 100);
        TVector<i32> b(100);
        FillWithRandomNumbers(b.data(), 239, 100);

        for (ui32 i = 0; i < 30; ++i) {
            for (ui32 length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT_EQUAL(DotProduct(a.data() + i, b.data() + i, length), (SimpleDotProduct<i64, i32>(a.data() + i, b.data() + i, length)));
                UNIT_ASSERT_EQUAL(DotProductSlow(a.data() + i, b.data() + i, length), (SimpleDotProduct<i64, i32>(a.data() + i, b.data() + i, length)));
            }
        }
    }

    Y_UNIT_TEST(TestDotProductf) {
        TVector<float> a(100);
        FillWithRandomFloats(a.data(), 179, 100);
        TVector<float> b(100);
        FillWithRandomFloats(b.data(), 239, 100);

        for (ui32 i = 0; i < 30; ++i) {
            for (ui32 length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT(std::fabs(DotProduct(a.data() + i, b.data() + i, length) - (SimpleDotProduct<float, float>(a.data() + i, b.data() + i, length))) < EPSILON);
                UNIT_ASSERT(std::fabs(DotProductSlow(a.data() + i, b.data() + i, length) - (SimpleDotProduct<float, float>(a.data() + i, b.data() + i, length))) < EPSILON);
            }
        }
    }

    Y_UNIT_TEST(TestL2NormSqaredf) {
        TVector<float> a(100);
        FillWithRandomFloats(a.data(), 179, 100);
        TVector<float> b(100);
        FillWithRandomFloats(b.data(), 239, 100);

        for (ui32 i = 0; i < 30; ++i) {
            for (ui32 length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT(std::fabs(L2NormSquared(a.data() + i, length) - DotProductSlow(a.data() + i, a.data() + i, length)) < EPSILON);
                UNIT_ASSERT(std::fabs(L2NormSquared(b.data() + i, length) - DotProductSlow(b.data() + i, b.data() + i, length)) < EPSILON);
            }
        }
    }

    Y_UNIT_TEST(TestDotProductd) {
        TVector<double> a(100);
        FillWithRandomFloats(a.data(), 179, 100);
        TVector<double> b(100);
        FillWithRandomFloats(b.data(), 239, 100);

        for (ui32 i = 0; i < 30; ++i) {
            for (ui32 length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT(std::fabs(DotProduct(a.data() + i, b.data() + i, length) - (SimpleDotProduct<double, double>(a.data() + i, b.data() + i, length))) < EPSILON);
                UNIT_ASSERT(std::fabs(DotProductSlow(a.data() + i, b.data() + i, length) - (SimpleDotProduct<double, double>(a.data() + i, b.data() + i, length))) < EPSILON);
            }
        }
    }

    Y_UNIT_TEST(TestCombinedDotProductf) {
        TVector<float> a(100);
        FillWithRandomFloats(a.data(), 179, 100);
        TVector<float> b(100);
        FillWithRandomFloats(b.data(), 239, 100);

        auto simple3WayProduct = [](const float* l, const float* r, ui32 length) -> TTriWayDotProduct<float> {
            return {
                SimpleDotProduct<float, float>(l, l, length),
                SimpleDotProduct<float, float>(l, r, length),
                SimpleDotProduct<float, float>(r, r, length)
            };
        };
        auto cosine = [](const auto p) {
            return p.LR / sqrt(p.LL * p.RR);
        };

        for (ui32 i = 0; i < 30; ++i) {
            for (ui32 length = 1; length + i + 1 < a.size(); ++length) {
                const TString testCaseExpl = TStringBuilder() << "i = " << i << "; length = " << length;
                {
                    const float c1 = cosine(TriWayDotProduct(a.data() + i, b.data() + i, length));
                    const float c2 = cosine(simple3WayProduct(a.data() + i, b.data() + i, length));
                    UNIT_ASSERT_DOUBLES_EQUAL_C(c1, c2, EPSILON, testCaseExpl);
                }
                {
                    // Left
                    auto cpl = TriWayDotProduct(a.data() + i, b.data() + i, length, ETriWayDotProductComputeMask::Left);
                    auto cnl = simple3WayProduct(a.data() + i, b.data() + i, length);
                    UNIT_ASSERT_DOUBLES_EQUAL(cpl.RR, 1.0, EPSILON);
                    cpl.RR = 1;
                    cnl.RR = 1;
                    UNIT_ASSERT_DOUBLES_EQUAL_C(cosine(cpl), cosine(cnl), EPSILON, testCaseExpl);
                }
                {
                    // Right
                    auto cpr = TriWayDotProduct(a.data() + i, b.data() + i, length, ETriWayDotProductComputeMask::Right);
                    auto cnr = simple3WayProduct(a.data() + i, b.data() + i, length);
                    UNIT_ASSERT_DOUBLES_EQUAL(cpr.LL, 1.0, EPSILON);
                    cpr.LL = 1;
                    cnr.LL = 1;
                    UNIT_ASSERT_DOUBLES_EQUAL_C(cosine(cpr), cosine(cnr), EPSILON, testCaseExpl);
                }
            }
        }
    }

    Y_UNIT_TEST(TestDotProductZeroLength) {
        UNIT_ASSERT_EQUAL(DotProduct(static_cast<const i8*>(nullptr), nullptr, 0), 0);
        UNIT_ASSERT_EQUAL(DotProduct(static_cast<const ui8*>(nullptr), nullptr, 0), 0);
        UNIT_ASSERT_EQUAL(DotProduct(static_cast<const i32*>(nullptr), nullptr, 0), 0);
        UNIT_ASSERT(std::abs(DotProduct(static_cast<const float*>(nullptr), nullptr, 0)) < EPSILON);
        UNIT_ASSERT(std::abs(DotProduct(static_cast<const double*>(nullptr), nullptr, 0)) < EPSILON);
        UNIT_ASSERT_EQUAL(DotProductSlow(static_cast<const i8*>(nullptr), nullptr, 0), 0);
        UNIT_ASSERT_EQUAL(DotProductSlow(static_cast<const ui8*>(nullptr), nullptr, 0), 0);
        UNIT_ASSERT_EQUAL(DotProductSlow(static_cast<const i32*>(nullptr), nullptr, 0), 0);
        UNIT_ASSERT(std::abs(DotProductSlow(static_cast<const float*>(nullptr), nullptr, 0)) < EPSILON);
        UNIT_ASSERT(std::abs(DotProductSlow(static_cast<const double*>(nullptr), nullptr, 0)) < EPSILON);
    }

    Y_UNIT_TEST(TestDotProductFloatStability) {
        TVector<float> a(1003);
        FillWithRandomFloats(a.data(), 179, a.size());
        TVector<float> b(1003);
        FillWithRandomFloats(b.data(), 239, b.size());

        float res = DotProduct(a.data(), b.data(), a.size());

        for (ui32 i = 0; i < 30; ++i)
            UNIT_ASSERT_VALUES_EQUAL(DotProduct(a.data(), b.data(), a.size()), res);

#ifdef ARCADIA_SSE
        UNIT_ASSERT_VALUES_EQUAL(ToString(res), "250.502");
#endif
    }

    Y_UNIT_TEST(TestDotProductDoubleStability) {
        TVector<double> a(1003);
        FillWithRandomFloats(a.data(), 13133, a.size());
        TVector<double> b(1003);
        FillWithRandomFloats(b.data(), 1121, b.size());

        double res = DotProduct(a.data(), b.data(), a.size());

        for (ui32 i = 0; i < 30; ++i)
            UNIT_ASSERT_VALUES_EQUAL(DotProduct(a.data(), b.data(), a.size()), res);

#ifdef ARCADIA_SSE
        UNIT_ASSERT_VALUES_EQUAL(ToString(res), "235.7826026");
#endif
    }

    Y_UNIT_TEST(TestDotProductCharStability) {
        TVector<i8> a(1003);
        FillWithRandomNumbers(a.data(), 1079, a.size());
        TVector<i8> b(1003);
        FillWithRandomNumbers(b.data(), 2139, b.size());

        ui32 res = DotProduct(a.data(), b.data(), a.size());

        for (ui32 i = 0; i < 30; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(DotProduct(a.data(), b.data(), a.size()), res);
            UNIT_ASSERT_VALUES_EQUAL(DotProductSlow(a.data(), b.data(), a.size()), res);
        }

        UNIT_ASSERT_VALUES_EQUAL(res, 90928);
    }

    Y_UNIT_TEST(TestDotProductCharStabilityU) {
        TVector<ui8> a(1003);
        FillWithRandomNumbers(a.data(), 1079, a.size());
        TVector<ui8> b(1003);
        FillWithRandomNumbers(b.data(), 2139, b.size());

        ui32 res = DotProduct(a.data(), b.data(), a.size());

        for (ui32 i = 0; i < 30; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(DotProduct(a.data(), b.data(), a.size()), res);
            UNIT_ASSERT_VALUES_EQUAL(DotProductSlow(a.data(), b.data(), a.size()), res);
        }

        UNIT_ASSERT_VALUES_EQUAL(res, 16420179);
    }
}
