#include "dot_product.h"
#include "dot_product_simple.h"
#include "dot_product_avx2.h"
#include "dot_product_vnni.h"

#include <library/cpp/testing/unittest/registar.h>

#include <library/cpp/sse/sse.h>
#include <util/generic/vector.h>
#include <util/random/fast.h>
#include <util/system/cpu_id.h>

#include <cmath>

Y_UNIT_TEST_SUITE(TDocProductTestSuite) {
    const double EPSILON = 0.00001;
    template <class Num>
    void FillWithRandomNumbers(Num * dst, int seed, size_t length) {
        TReallyFastRng32 Rnd(seed);
        Num maxNum = ~Num(0);
        for (size_t i = 0; i < length; ++i) {
            dst[i] = Rnd.Uniform(maxNum);
        }
    }

    template <class Num>
    void FillWithRandomFloats(Num * dst, int seed, size_t length) {
        TReallyFastRng32 Rnd(seed);
        for (size_t i = 0; i < length; ++i) {
            dst[i] = Rnd.GenRandReal1();
        }
    }

    template <class Res, class Int>
    Res SimpleDotProduct(const Int* lhs, const Int* rhs, size_t length) {
        Res sum = 0;
        for (size_t i = 0; i < length; ++i) {
            sum += static_cast<Res>(lhs[i]) * static_cast<Res>(rhs[i]);
        }
        return sum;
    }

    template <class Res, class TLhs, class TRhs>
    Res SimpleDotProductMixed(const TLhs* lhs, const TRhs* rhs, size_t length) {
        Res sum = 0;
        for (size_t i = 0; i < length; ++i) {
            sum += static_cast<Res>(lhs[i]) * static_cast<Res>(rhs[i]);
        }
        return sum;
    }

    bool HaveVnniDotProduct() {
        return NX86::HaveAVX2() && NX86::HaveFMA() && NX86::HaveAVX512BW() && NX86::HaveAVX512VNNI();
    }

    Y_UNIT_TEST(TestDotProduct8) {
        TVector<i8> a(100);
        FillWithRandomNumbers(a.data(), 179, 100);
        TVector<i8> b(100);
        FillWithRandomNumbers(b.data(), 239, 100);

        const bool haveVnni = HaveVnniDotProduct();
        for (size_t i = 0; i < 30; ++i) {
            for (size_t length = 1; length + i + 1 < a.size(); ++length) {
                const i32 expected = SimpleDotProduct<i32, i8>(a.data() + i, b.data() + i, length);
                UNIT_ASSERT_EQUAL(DotProduct(a.data() + i, b.data() + i, length), expected);
                UNIT_ASSERT_EQUAL(DotProductSimple(a.data() + i, b.data() + i, length), expected);
                if (haveVnni) {
                    UNIT_ASSERT_EQUAL(DotProductVnni(a.data() + i, b.data() + i, length), expected);
                }
            }
        }
    }

    Y_UNIT_TEST(TestDotProduct8VnniEdges) {
        if (!HaveVnniDotProduct()) {
            return;
        }

        TVector<i8> a(512);
        TVector<i8> b(512);
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<i8>((i * 37) % 256 - 128);
            b[i] = static_cast<i8>((i * 53 + 17) % 256 - 128);
        }

        for (size_t offset = 0; offset < 16; ++offset) {
            for (size_t length = 0; length + offset <= a.size(); ++length) {
                UNIT_ASSERT_EQUAL(
                    DotProductVnni(a.data() + offset, b.data() + offset, length),
                    (SimpleDotProduct<i32, i8>(a.data() + offset, b.data() + offset, length)));
            }
        }
    }

    Y_UNIT_TEST(TestDotProduct8u) {
        TVector<ui8> a(100);
        FillWithRandomNumbers(a.data(), 179, 100);
        TVector<ui8> b(100);
        FillWithRandomNumbers(b.data(), 239, 100);

        for (size_t i = 0; i < 30; ++i) {
            for (size_t length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT_EQUAL(DotProduct(a.data() + i, b.data() + i, length), (SimpleDotProduct<ui32, ui8>(a.data() + i, b.data() + i, length)));
                UNIT_ASSERT_EQUAL(DotProductSimple(a.data() + i, b.data() + i, length), (SimpleDotProduct<ui32, ui8>(a.data() + i, b.data() + i, length)));
            }
        }
    }

    Y_UNIT_TEST(TestDotProduct32) {
        TVector<i32> a(100);
        FillWithRandomNumbers(a.data(), 179, 100);
        TVector<i32> b(100);
        FillWithRandomNumbers(b.data(), 239, 100);

        for (size_t i = 0; i < 30; ++i) {
            for (size_t length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT_EQUAL(DotProduct(a.data() + i, b.data() + i, length), (SimpleDotProduct<i64, i32>(a.data() + i, b.data() + i, length)));
                UNIT_ASSERT_EQUAL(DotProductSimple(a.data() + i, b.data() + i, length), (SimpleDotProduct<i64, i32>(a.data() + i, b.data() + i, length)));
            }
        }
    }

    Y_UNIT_TEST(TestDotProductf) {
        TVector<float> a(100);
        FillWithRandomFloats(a.data(), 179, 100);
        TVector<float> b(100);
        FillWithRandomFloats(b.data(), 239, 100);

        for (size_t i = 0; i < 30; ++i) {
            for (size_t length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT(std::fabs(DotProduct(a.data() + i, b.data() + i, length) - (SimpleDotProduct<float, float>(a.data() + i, b.data() + i, length))) < EPSILON);
                UNIT_ASSERT(std::fabs(DotProductSimple(a.data() + i, b.data() + i, length) - (SimpleDotProduct<float, float>(a.data() + i, b.data() + i, length))) < EPSILON);
            }
        }
    }

    Y_UNIT_TEST(TestDotProductFloatI8) {
        TVector<float> floats(100);
        FillWithRandomFloats(floats.data(), 179, 100);
        TVector<i8> bytes(100);
        FillWithRandomNumbers(bytes.data(), 239, 100);

        constexpr double MIXED_EPSILON = 1e-4;
        for (size_t i = 0; i < 30; ++i) {
            for (size_t length = 1; length + i + 1 < floats.size(); ++length) {
                const float expected = SimpleDotProductMixed<float, float, i8>(floats.data() + i, bytes.data() + i, length);
                UNIT_ASSERT(std::fabs(DotProduct(floats.data() + i, bytes.data() + i, length) - expected) < MIXED_EPSILON);
                UNIT_ASSERT(std::fabs(DotProductSimple(floats.data() + i, bytes.data() + i, length) - expected) < EPSILON);
                if (NX86::HaveAVX2() && NX86::HaveFMA()) {
                    UNIT_ASSERT(std::fabs(DotProductFloatI8Avx2(floats.data() + i, bytes.data() + i, length) - expected) < MIXED_EPSILON);
                }
            }
        }
    }

    Y_UNIT_TEST(TestTriWayDotProductFloatI8) {
        TVector<float> floats(100);
        FillWithRandomFloats(floats.data(), 179, 100);
        TVector<i8> bytes(100);
        FillWithRandomNumbers(bytes.data(), 239, 100);

        constexpr double MIXED_EPSILON = 1e-4;
        for (size_t i = 0; i < 30; ++i) {
            for (size_t length = 1; length + i + 1 < floats.size(); ++length) {
                const auto expected = TriWayDotProductFloatI8Simple(floats.data() + i, bytes.data() + i, length);
                const auto actual = TriWayDotProduct(floats.data() + i, bytes.data() + i, length);
                UNIT_ASSERT_DOUBLES_EQUAL(expected.LL, actual.LL, MIXED_EPSILON);
                UNIT_ASSERT_DOUBLES_EQUAL(expected.LR, actual.LR, MIXED_EPSILON);
                UNIT_ASSERT_DOUBLES_EQUAL(expected.RR, actual.RR, MIXED_EPSILON);
                if (NX86::HaveAVX2() && NX86::HaveFMA()) {
                    const auto avx2 = TriWayDotProductFloatI8Avx2(floats.data() + i, bytes.data() + i, length);
                    UNIT_ASSERT_DOUBLES_EQUAL(expected.LL, avx2.LL, MIXED_EPSILON);
                    UNIT_ASSERT_DOUBLES_EQUAL(expected.LR, avx2.LR, MIXED_EPSILON);
                    UNIT_ASSERT_DOUBLES_EQUAL(expected.RR, avx2.RR, MIXED_EPSILON);
                }
            }
        }
    }

    Y_UNIT_TEST(TestTriWayDotProductI8) {
        TVector<i8> a(100);
        FillWithRandomNumbers(a.data(), 179, 100);
        TVector<i8> b(100);
        FillWithRandomNumbers(b.data(), 239, 100);

        for (size_t i = 0; i < 30; ++i) {
            for (size_t length = 1; length + i + 1 < a.size(); ++length) {
                const auto expected = TriWayDotProductI8Simple(a.data() + i, b.data() + i, length);
                const auto actual = TriWayDotProduct(a.data() + i, b.data() + i, length);
                UNIT_ASSERT_EQUAL(expected.LL, actual.LL);
                UNIT_ASSERT_EQUAL(expected.LR, actual.LR);
                UNIT_ASSERT_EQUAL(expected.RR, actual.RR);
                if (NX86::HaveAVX2() && NX86::HaveFMA()) {
                    const auto avx2 = TriWayDotProductI8Avx2(a.data() + i, b.data() + i, length);
                    UNIT_ASSERT_EQUAL(expected.LL, avx2.LL);
                    UNIT_ASSERT_EQUAL(expected.LR, avx2.LR);
                    UNIT_ASSERT_EQUAL(expected.RR, avx2.RR);
                }
            }
        }
    }

    Y_UNIT_TEST(TestL2NormSqaredf) {
        TVector<float> a(100);
        FillWithRandomFloats(a.data(), 179, 100);
        TVector<float> b(100);
        FillWithRandomFloats(b.data(), 239, 100);

        for (size_t i = 0; i < 30; ++i) {
            for (size_t length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT(std::fabs(L2NormSquared(a.data() + i, length) - DotProductSimple(a.data() + i, a.data() + i, length)) < EPSILON);
                UNIT_ASSERT(std::fabs(L2NormSquared(b.data() + i, length) - DotProductSimple(b.data() + i, b.data() + i, length)) < EPSILON);
            }
        }
    }

    Y_UNIT_TEST(TestDotProductd) {
        TVector<double> a(100);
        FillWithRandomFloats(a.data(), 179, 100);
        TVector<double> b(100);
        FillWithRandomFloats(b.data(), 239, 100);

        for (size_t i = 0; i < 30; ++i) {
            for (size_t length = 1; length + i + 1 < a.size(); ++length) {
                UNIT_ASSERT(std::fabs(DotProduct(a.data() + i, b.data() + i, length) - (SimpleDotProduct<double, double>(a.data() + i, b.data() + i, length))) < EPSILON);
                UNIT_ASSERT(std::fabs(DotProductSimple(a.data() + i, b.data() + i, length) - (SimpleDotProduct<double, double>(a.data() + i, b.data() + i, length))) < EPSILON);
            }
        }
    }

    Y_UNIT_TEST(TestCombinedDotProductf) {
        TVector<float> a(100);
        FillWithRandomFloats(a.data(), 179, 100);
        TVector<float> b(100);
        FillWithRandomFloats(b.data(), 239, 100);

        auto simple3WayProduct = [](const float* l, const float* r, size_t length) -> TTriWayDotProduct<float> {
            return {
                SimpleDotProduct<float, float>(l, l, length),
                SimpleDotProduct<float, float>(l, r, length),
                SimpleDotProduct<float, float>(r, r, length)
            };
        };
        auto cosine = [](const auto p) {
            return p.LR / sqrt(p.LL * p.RR);
        };

        for (size_t i = 0; i < 30; ++i) {
            for (size_t length = 1; length + i + 1 < a.size(); ++length) {
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
        UNIT_ASSERT(std::abs(DotProduct(static_cast<const float*>(nullptr), static_cast<const float*>(nullptr), 0)) < EPSILON);
        UNIT_ASSERT(std::abs(DotProduct(static_cast<const float*>(nullptr), static_cast<const i8*>(nullptr), 0)) < EPSILON);
        UNIT_ASSERT(std::abs(DotProduct(static_cast<const double*>(nullptr), nullptr, 0)) < EPSILON);
        UNIT_ASSERT_EQUAL(DotProductSimple(static_cast<const i8*>(nullptr), nullptr, 0), 0);
        UNIT_ASSERT_EQUAL(DotProductSimple(static_cast<const ui8*>(nullptr), nullptr, 0), 0);
        UNIT_ASSERT_EQUAL(DotProductSimple(static_cast<const i32*>(nullptr), nullptr, 0), 0);
        UNIT_ASSERT(std::abs(DotProductSimple(static_cast<const float*>(nullptr), static_cast<const float*>(nullptr), 0)) < EPSILON);
        UNIT_ASSERT(std::abs(DotProductSimple(static_cast<const float*>(nullptr), static_cast<const i8*>(nullptr), 0)) < EPSILON);
        UNIT_ASSERT(std::abs(DotProductSimple(static_cast<const double*>(nullptr), nullptr, 0)) < EPSILON);
    }

    Y_UNIT_TEST(TestDotProductFloatStability) {
        TVector<float> a(1003);
        FillWithRandomFloats(a.data(), 179, a.size());
        TVector<float> b(1003);
        FillWithRandomFloats(b.data(), 239, b.size());

        float res = DotProduct(a.data(), b.data(), a.size());

        for (size_t i = 0; i < 30; ++i)
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

        for (size_t i = 0; i < 30; ++i)
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

        for (size_t i = 0; i < 30; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(DotProduct(a.data(), b.data(), a.size()), res);
            UNIT_ASSERT_VALUES_EQUAL(DotProductSimple(a.data(), b.data(), a.size()), res);
        }

        UNIT_ASSERT_VALUES_EQUAL(res, 90928);
    }

    Y_UNIT_TEST(TestDotProductCharStabilityU) {
        TVector<ui8> a(1003);
        FillWithRandomNumbers(a.data(), 1079, a.size());
        TVector<ui8> b(1003);
        FillWithRandomNumbers(b.data(), 2139, b.size());

        ui32 res = DotProduct(a.data(), b.data(), a.size());

        for (size_t i = 0; i < 30; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(DotProduct(a.data(), b.data(), a.size()), res);
            UNIT_ASSERT_VALUES_EQUAL(DotProductSimple(a.data(), b.data(), a.size()), res);
        }

        UNIT_ASSERT_VALUES_EQUAL(res, 16420179);
    }

    Y_UNIT_TEST(TestDotProductUI4Manual) {
        static ui8 a[4] = {1 + (3 << 4), 15 + (8 << 4), 0 + (5 << 4), 3 + (1 << 4)};
        static ui8 b[4] = {2 + (4 << 4), 1 + (8 << 4), 7 + (0 << 4), 1 + (4 << 4)};
        UNIT_ASSERT_VALUES_EQUAL(DotProductUI4Simple(a, b, 4), 2 + 12 + 15 + 64 + 0 + 0 + 3 + 4);
    }
}
