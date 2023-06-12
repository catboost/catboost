#include "utility.h"
#include "ymath.h"

#include <library/cpp/testing/unittest/registar.h>

// DO_NOT_STYLE

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

struct TUnorderedTag {
    TStringBuf Tag;
};

static bool operator<(const TUnorderedTag, const TUnorderedTag) {
    return false;
}

static bool operator>(const TUnorderedTag, const TUnorderedTag) = delete;

Y_UNIT_TEST_SUITE(TUtilityTest) {

    Y_UNIT_TEST(TestSwapPrimitive) {
        int i = 0;
        int j = 1;

        DoSwap(i, j);

        UNIT_ASSERT_EQUAL(i, 1);
        UNIT_ASSERT_EQUAL(j, 0);
    }

    Y_UNIT_TEST(TestSwapClass) {
        TTest i(0);
        TTest j(1);

        DoSwap(i, j);

        UNIT_ASSERT_EQUAL(i.Val, 1);
        UNIT_ASSERT_EQUAL(j.Val, 0);
    }

    Y_UNIT_TEST(TestMaxMin) {
        static_assert(Min(10, 3, 8) == 3, "Min doesn't work");
        static_assert(Max(10, 3, 8) == 10, "Max doesn't work");
        UNIT_ASSERT_EQUAL(Min(10, 3, 8), 3);
        UNIT_ASSERT_EQUAL(Max(3.5, 4.2, 8.1, 99.025, 0.33, 29.0), 99.025);

        UNIT_ASSERT_VALUES_EQUAL(Min(TUnorderedTag{"first"}, TUnorderedTag{"second"}).Tag, "first");
        UNIT_ASSERT_VALUES_EQUAL(Max(TUnorderedTag{"first"}, TUnorderedTag{"second"}).Tag, "first");
        UNIT_ASSERT_VALUES_EQUAL(Min(TUnorderedTag{"first"}, TUnorderedTag{"second"}, TUnorderedTag{"third"}).Tag, "first");
        UNIT_ASSERT_VALUES_EQUAL(Max(TUnorderedTag{"first"}, TUnorderedTag{"second"}, TUnorderedTag{"third"}).Tag, "first");
    }

    Y_UNIT_TEST(TestMean) {
        UNIT_ASSERT_EQUAL(Mean(5), 5);
        UNIT_ASSERT_EQUAL(Mean(1, 2, 3), 2);
        UNIT_ASSERT_EQUAL(Mean(6, 5, 4), 5);
        UNIT_ASSERT_EQUAL(Mean(1, 2), 1.5);
        UNIT_ASSERT(Abs(Mean(1., 2., 7.5) - 3.5) < std::numeric_limits<double>::epsilon());
    }

    Y_UNIT_TEST(TestZeroInitWithDefaultZeros) {
        struct TStructWithPaddingBytes: public TZeroInit<TStructWithPaddingBytes> {
            TStructWithPaddingBytes()
                : TZeroInit<TStructWithPaddingBytes>() {
            }
            bool Field1_ = static_cast<bool>(0);
            // here between Field1_ and Field2_ will be padding bytes
            i64 Field2_ = 0;
        };

        TStructWithPaddingBytes foo{};

        // all bytes must be zeroes, and MSAN will not complain about reading from padding bytes
        const char* const fooPtr = (char*)&foo;
        for (size_t i = 0; i < sizeof(TStructWithPaddingBytes); ++i) {
            const char byte = fooPtr[i];
            UNIT_ASSERT_EQUAL(byte, 0);
        }
    }

    Y_UNIT_TEST(TestZeroInitWithDefaultNonZeros) {
        struct TStructWithPaddingBytes: public TZeroInit<TStructWithPaddingBytes> {
            TStructWithPaddingBytes()
                : TZeroInit<TStructWithPaddingBytes>() {
            }
            bool Field1_ = true;
            // here between Field1_ and Field2_ will be padding bytes
            i64 Field2_ = 100500;
        };

        TStructWithPaddingBytes foo{};

        // check that default values are set correctly
        UNIT_ASSERT_EQUAL(foo.Field1_, true);
        UNIT_ASSERT_EQUAL(foo.Field2_, 100500);

        const char* const fooPtr = (char*)&foo;
        // just reading all bytes, and MSAN must not complain about reading padding bytes
        for (size_t i = 0; i < sizeof(TStructWithPaddingBytes); ++i) {
            const char byte = fooPtr[i];
            UNIT_ASSERT_EQUAL(byte, byte);
        }
    }

    Y_UNIT_TEST(TestClampValNoClamp) {
        double val = 2;
        double lo = 1;
        double hi = 3;
        const double& clamped = ClampVal(val, lo, hi);
        UNIT_ASSERT_EQUAL(clamped, val);
        UNIT_ASSERT_EQUAL(&clamped, &val);
    }

    Y_UNIT_TEST(TestClampValLo) {
        double val = 2;
        double lo = 3;
        double hi = 4;
        const double& clamped = ClampVal(val, lo, hi);
        UNIT_ASSERT_EQUAL(clamped, lo);
        UNIT_ASSERT_EQUAL(&clamped, &lo);
    }

    Y_UNIT_TEST(TestClampValHi) {
        double val = 4;
        double lo = 3;
        double hi = 2;
        const double& clamped = ClampVal(val, lo, hi);
        UNIT_ASSERT_EQUAL(clamped, hi);
        UNIT_ASSERT_EQUAL(&clamped, &hi);
    }

    Y_UNIT_TEST(TestSecureZero) {
        constexpr size_t checkSize = 128;
        char test[checkSize];

        // fill with garbage
        for (size_t i = 0; i < checkSize; ++i) {
            test[i] = i;
        }

        SecureZero(test, checkSize);

        for (size_t i = 0; i < checkSize; ++i) {
            UNIT_ASSERT_EQUAL(test[i], 0);
        }
    }

    Y_UNIT_TEST(TestSecureZeroTemplate) {
        constexpr size_t checkSize = 128;
        char test[checkSize];

        // fill with garbage
        for (size_t i = 0; i < checkSize; ++i) {
            test[i] = i;
        }

        SecureZero(test);

        for (size_t i = 0; i < checkSize; ++i) {
            UNIT_ASSERT_EQUAL(test[i], 0);
        }
    }
}
