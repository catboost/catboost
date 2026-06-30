#include "fast.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TTestFastRng) {
    Y_UNIT_TEST(Test1) {
        TFastRng32 rng1(17, 0);
        TReallyFastRng32 rng2(17);

        for (size_t i = 0; i < 1000; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(rng1.GenRand(), rng2.GenRand());
        }
    }

    static ui64 R1[] = {
        37,
        43,
        76,
        17,
        12,
        87,
        60,
        4,
        83,
        47,
        57,
        81,
        28,
        45,
        66,
        74,
        18,
        17,
        18,
        75,
    };

    Y_UNIT_TEST(Test2) {
        TFastRng64 rng(0, 1, 2, 3);

        for (auto& i : R1) {
            UNIT_ASSERT_VALUES_EQUAL(rng.Uniform(100u), i);
        }
    }

    Y_UNIT_TEST(TestAdvance) {
        TReallyFastRng32 rng1(17);
        TReallyFastRng32 rng2(17);
        for (size_t i = 0; i < 100; i++) {
            rng1.GenRand();
        }
        rng2.Advance(100);
        UNIT_ASSERT_VALUES_EQUAL(rng1.GenRand(), rng2.GenRand());

        TFastRng64 rng3(0, 1, 2, 3);
        TFastRng64 rng4(0, 1, 2, 3);
        for (size_t i = 0; i < 100; i++) {
            rng3.GenRand();
        }
        rng4.Advance(100);
        UNIT_ASSERT_VALUES_EQUAL(rng3.GenRand(), rng4.GenRand());
    }

    Y_UNIT_TEST(TestAdvanceBoundaries) {
        TReallyFastRng32 rng1(17);
        TReallyFastRng32 rng2(17);
        TReallyFastRng32 rng3(17);
        rng2.Advance(0);
        rng3.Advance(1);
        UNIT_ASSERT_VALUES_EQUAL(rng1.GenRand(), rng2.GenRand());
        UNIT_ASSERT_VALUES_EQUAL(rng1.GenRand(), rng3.GenRand());
    }

    Y_UNIT_TEST(TestCopy) {
        TReallyFastRng32 r1(1);
        TReallyFastRng32 r2(2);

        UNIT_ASSERT_VALUES_UNEQUAL(r1.GenRand(), r2.GenRand());

        r2 = r1;

        UNIT_ASSERT_VALUES_EQUAL(r1.GenRand(), r2.GenRand());
    }

    Y_UNIT_TEST(Test3) {
        TFastRng64 rng(17);

        UNIT_ASSERT_VALUES_EQUAL(rng.GenRand(), ULL(14895365814383052362));
    }

    Y_UNIT_TEST(TestCompile) {
        TFastRng<ui32> rng1(1);
        TFastRng<ui64> rng2(2);
        TFastRng<size_t> rng3(3);

        rng1.GenRand();
        rng2.GenRand();
        rng3.GenRand();
    }

    const char* RNG_DATA = "At the top of the department store I,"
                           "I bought a fur coat with fur I"
                           "But apparently I made a blunder here -"
                           "Doha does not warm ... Absolutely.";

    Y_UNIT_TEST(TestStreamCtor1) {
        TMemoryInput mi(RNG_DATA, strlen(RNG_DATA));
        TFastRng<ui32> rng(mi);

        UNIT_ASSERT_VALUES_EQUAL(rng.GenRand(), 1449109131u);
    }

    Y_UNIT_TEST(TestStreamCtor2) {
        TMemoryInput mi(RNG_DATA, strlen(RNG_DATA));
        TFastRng<ui64> rng(mi);

        UNIT_ASSERT_VALUES_EQUAL(rng.GenRand(), ULL(6223876579076085114));
    }
} // Y_UNIT_TEST_SUITE(TTestFastRng)
