#include "bitops.h"
#include "ymath.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/output.h>
#include <util/datetime/cputimer.h>

#include <limits>

template <class T>
static inline T SlowClp2(T t) noexcept {
    Y_ASSERT(t > 0);

    T ret = 1;

    while (ret < t) {
        ret *= 2;
    }

    return ret;
}

class TMathTest: public TTestBase {
    UNIT_TEST_SUITE(TMathTest);
    UNIT_TEST(TestClp2)
    UNIT_TEST(TestClpSimple)
    UNIT_TEST(TestSqr)
    UNIT_TEST(TestLog2)
    UNIT_TEST(ValueBitCount)
    UNIT_TEST(TestErf);
    UNIT_TEST(TestLogGamma);
    UNIT_TEST(TestIsValidFloat);
    UNIT_TEST(TestAbs);
    UNIT_TEST(TestPower);
    UNIT_TEST(TestSigmoid);
    UNIT_TEST(TestCeilDiv);
    UNIT_TEST_SUITE_END();

private:
    void TestClp2();
    void TestSqr();
    void TestErf();
    void TestLogGamma();
    void TestAbs();
    void TestPower();
    void TestSigmoid();
    void TestCeilDiv();

    inline void TestIsValidFloat() {
        UNIT_ASSERT(IsValidFloat(-Max<double>() / 2.));
    }

    inline void TestClpSimple() {
        UNIT_ASSERT_EQUAL(FastClp2<ui32>(12), 16);
        UNIT_ASSERT_EQUAL(FastClp2<ui16>(11), 16);
        UNIT_ASSERT_EQUAL(FastClp2<ui8>(10), 16);

        UNIT_ASSERT_EQUAL(FastClp2<ui32>(15), 16);
        UNIT_ASSERT_EQUAL(FastClp2<ui32>(16), 16);
        UNIT_ASSERT_EQUAL(FastClp2<ui32>(17), 32);
    }

    inline void TestLog2() {
        UNIT_ASSERT_DOUBLES_EQUAL(Log2(2.0), 1.0, 1e-10);
        UNIT_ASSERT_DOUBLES_EQUAL(Log2(2ull), 1.0, 1e-10);
        UNIT_ASSERT_DOUBLES_EQUAL(Log2(2.0f), 1.0f, 1e-7f);
    }

    inline void ValueBitCount() {
        UNIT_ASSERT_VALUES_EQUAL(GetValueBitCount(1), 1u);
        UNIT_ASSERT_VALUES_EQUAL(GetValueBitCount(2), 2u);
        UNIT_ASSERT_VALUES_EQUAL(GetValueBitCount(3), 2u);
        UNIT_ASSERT_VALUES_EQUAL(GetValueBitCount(257), 9u);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TMathTest);

void TMathTest::TestSqr() {
    UNIT_ASSERT_EQUAL(Sqr(2), 4);
    UNIT_ASSERT_EQUAL(Sqr(2.0), 4.0);
}

void TMathTest::TestClp2() {
    for (ui8 i = 1; i < 127; ++i) {
        UNIT_ASSERT_EQUAL(SlowClp2(i), FastClp2(i));
    }

    for (ui16 i = 1; i < 255; ++i) {
        UNIT_ASSERT_EQUAL(SlowClp2(i), FastClp2(i));
    }

    for (ui32 i = 1; i < 255; ++i) {
        UNIT_ASSERT_EQUAL(SlowClp2(i), FastClp2(i));
    }

    for (ui64 i = 1; i < 255; ++i) {
        UNIT_ASSERT_EQUAL(SlowClp2(i), FastClp2(i));
    }

    if (0) {
        {
            TFuncTimer timer("fast");
            size_t ret = 0;

            for (size_t i = 0; i < 10000000; ++i) {
                ret += FastClp2(i);
            }

            Cerr << ret << Endl;
        }

        {
            TFuncTimer timer("slow");
            size_t ret = 0;

            for (size_t i = 0; i < 10000000; ++i) {
                ret += SlowClp2(i);
            }

            Cerr << ret << Endl;
        }
    }
}

void TMathTest::TestErf() {
    static const double a = -5.0;
    static const double b = 5.0;
    static const int n = 50;
    static const double step = (b - a) / n;

    static const double values[n + 1] = {
        -1.0000000, -1.0000000, -1.0000000, -1.0000000, -1.0000000,
        -1.0000000, -0.9999999, -0.9999996, -0.9999985, -0.9999940,
        -0.9999779, -0.9999250, -0.9997640, -0.9993115, -0.9981372,
        -0.9953223, -0.9890905, -0.9763484, -0.9522851, -0.9103140,
        -0.8427008, -0.7421010, -0.6038561, -0.4283924, -0.2227026,
        0.0000000,
        0.2227026, 0.4283924, 0.6038561, 0.7421010, 0.8427008,
        0.9103140, 0.9522851, 0.9763484, 0.9890905, 0.9953223,
        0.9981372, 0.9993115, 0.9997640, 0.9999250, 0.9999779,
        0.9999940, 0.9999985, 0.9999996, 0.9999999, 1.0000000,
        1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000};

    double x = a;
    for (int i = 0; i <= n; ++i, x += step) {
        double f = Erf(x);
        UNIT_ASSERT_DOUBLES_EQUAL(f, values[i], 1e-7);
    }
}

void TMathTest::TestLogGamma() {
    double curVal = 0.0;
    for (int i = 1; i <= 20; i++) {
        curVal += log((double)i);
        UNIT_ASSERT_DOUBLES_EQUAL(curVal, LogGamma((double)(i + 1)), 1e-6);
    }
    curVal = log(M_PI) / 2.0;
    for (int i = 1; i <= 20; i++) {
        UNIT_ASSERT_DOUBLES_EQUAL(curVal, LogGamma(i - 0.5), 1e-6);
        curVal += log(i - 0.5);
    }
}

void TMathTest::TestAbs() {
    UNIT_ASSERT_VALUES_EQUAL(Abs(1), 1);
    UNIT_ASSERT_VALUES_EQUAL(Abs(-1), 1);
    UNIT_ASSERT_VALUES_EQUAL(Abs(-1000000000000ll), 1000000000000ll);
    UNIT_ASSERT_VALUES_EQUAL(Abs(0), 0);
    UNIT_ASSERT_VALUES_EQUAL(Abs(1.0), 1.0);
    UNIT_ASSERT_VALUES_EQUAL(Abs(-1.0), 1.0);
    UNIT_ASSERT_VALUES_EQUAL(Abs(0.0), 0.0);
}

void TMathTest::TestPower() {
    UNIT_ASSERT_VALUES_EQUAL(Power(0, 0), 1);
    UNIT_ASSERT_VALUES_EQUAL(Power(-1, 1), -1);
    UNIT_ASSERT_VALUES_EQUAL(Power(-1, 2), 1);
    UNIT_ASSERT_VALUES_EQUAL(Power(2LL, 32), 1LL << 32);
    UNIT_ASSERT_DOUBLES_EQUAL(Power(0.0, 0), 1.0, 1e-9);
    UNIT_ASSERT_DOUBLES_EQUAL(Power(0.1, 3), 1e-3, 1e-9);
}

void TMathTest::TestSigmoid() {
    UNIT_ASSERT_EQUAL(Sigmoid(0.f), 0.5f);
    UNIT_ASSERT_EQUAL(Sigmoid(-5000.f), 0.0f);
    UNIT_ASSERT_EQUAL(Sigmoid(5000.f), 1.0f);

    UNIT_ASSERT_EQUAL(Sigmoid(0.), 0.5);
    UNIT_ASSERT_EQUAL(Sigmoid(-5000.), 0.0);
    UNIT_ASSERT_EQUAL(Sigmoid(5000.), 1.0);
}

void TMathTest::TestCeilDiv() {
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv<ui8>(2, 3), 1);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv<ui8>(3, 3), 1);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv<ui32>(12, 2), 6);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv<ui64>(10, 3), 4);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv<ui64>(0, 10), 0);

    // negative numbers
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv(0, -10), 0);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv(-1, 2), 0);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv(-1, -2), 1);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv(10, -5), -2);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv(-3, -4), 1);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv(-6, -4), 2);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv(-6, 4), -1);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv(-13, 4), -3);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv(-14, -4), 4);

    // check values close to overflow
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv<ui8>(255, 10), 26);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv<ui32>(std::numeric_limits<ui32>::max() - 3, std::numeric_limits<ui32>::max()), 1);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv<i32>(std::numeric_limits<i32>::max() - 3, std::numeric_limits<i32>::max()), 1);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv<i32>(std::numeric_limits<i32>::min(), std::numeric_limits<i32>::max()), -1);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv<i8>(std::numeric_limits<i8>::max(), std::numeric_limits<i8>::min() + 1), -1);
    UNIT_ASSERT_VALUES_EQUAL(CeilDiv<i64>(std::numeric_limits<i64>::max() - 2, -(std::numeric_limits<i64>::min() + 1)), 1);
}
