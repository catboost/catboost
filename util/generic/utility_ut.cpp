#include "utility.h"
#include "ymath.h"

#include <library/unittest/registar.h>

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


SIMPLE_UNIT_TEST_SUITE(TUtilityTest) {

    SIMPLE_UNIT_TEST(TestSwapPrimitive) {
        int i = 0;
        int j = 1;

        DoSwap(i, j);

        UNIT_ASSERT_EQUAL(i, 1);
        UNIT_ASSERT_EQUAL(j, 0);
    }

    SIMPLE_UNIT_TEST(TestSwapClass) {
        TTest i(0);
        TTest j(1);

        DoSwap(i, j);

        UNIT_ASSERT_EQUAL(i.Val, 1);
        UNIT_ASSERT_EQUAL(j.Val, 0);
    }

    SIMPLE_UNIT_TEST(TestMaxMin) {
        static_assert(Min(10, 3, 8) == 3, "Min doesn't work");
        static_assert(Max(10, 3, 8) == 10, "Max doesn't work");
        UNIT_ASSERT_EQUAL(Min(10, 3, 8), 3);
        UNIT_ASSERT_EQUAL(Max(3.5, 4.2, 8.1, 99.025, 0.33, 29.0), 99.025);
    }

    SIMPLE_UNIT_TEST(TestMean) {
        UNIT_ASSERT_EQUAL(Mean(5), 5);
        UNIT_ASSERT_EQUAL(Mean(1, 2, 3), 2);
        UNIT_ASSERT_EQUAL(Mean(6, 5, 4), 5);
        UNIT_ASSERT_EQUAL(Mean(1, 2), 1.5);
        UNIT_ASSERT(Abs(Mean(1., 2., 7.5) - 3.5) < std::numeric_limits<double>::epsilon());
    }

    SIMPLE_UNIT_TEST(TestZeroInitWithDefaultZeros) {
        struct TStructWithPaddingBytes : public TZeroInit<TStructWithPaddingBytes> {
            TStructWithPaddingBytes() : TZeroInit<TStructWithPaddingBytes>() {}
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

    SIMPLE_UNIT_TEST(TestZeroInitWithDefaultNonZeros) {
        struct TStructWithPaddingBytes : public TZeroInit<TStructWithPaddingBytes> {
            TStructWithPaddingBytes() : TZeroInit<TStructWithPaddingBytes>() {}
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
};
