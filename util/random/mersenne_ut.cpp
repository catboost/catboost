#include "mersenne.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/output.h>

#define UI32(x) x##ul

Y_UNIT_TEST_SUITE(TMersenneRndTest) {
    template <class T>
    inline void Test(const T* res, size_t len) {
        TMersenne<T> m;

        for (size_t i = 0; i < len; ++i) {
            UNIT_ASSERT_EQUAL(m.GenRand(), res[i]);
        }
    }

    Y_UNIT_TEST(Test32) {
        const ui32 res[] = {
            UI32(2325592414),
            UI32(482149846),
            UI32(4177211283),
            UI32(3872387439),
            UI32(1663027210),
            UI32(2005191859),
            UI32(666881213),
            UI32(3289399202),
            UI32(2514534568),
            UI32(3882134983),
        };

        Test<ui32>(res, Y_ARRAY_SIZE(res));
    }

    Y_UNIT_TEST(Test64) {
        const ui64 res[] = {
            ULL(13735441942630277712),
            ULL(10468394322237346228),
            ULL(5051557175812687784),
            ULL(8252857936377966838),
            ULL(4330799099585512958),
            ULL(327094098846779408),
            ULL(6143667654738189122),
            ULL(6387112078486713335),
            ULL(3862502196831460488),
            ULL(11231499428520958715),
        };

        Test<ui64>(res, Y_ARRAY_SIZE(res));
    }

    Y_UNIT_TEST(TestGenRand64) {
        TMersenne<ui32> rng;

        for (size_t i = 0; i < 100; ++i) {
            UNIT_ASSERT(rng.GenRand64() > ULL(12345678912));
        }
    }

    Y_UNIT_TEST(TestCopy32) {
        TMersenne<ui32> r1(1);
        TMersenne<ui32> r2(2);

        UNIT_ASSERT_VALUES_UNEQUAL(r1.GenRand(), r2.GenRand());

        r2 = r1;

        UNIT_ASSERT_VALUES_EQUAL(r1.GenRand(), r2.GenRand());
    }

    Y_UNIT_TEST(TestCopy64) {
        TMersenne<ui64> r1(1);
        TMersenne<ui64> r2(2);

        UNIT_ASSERT_VALUES_UNEQUAL(r1.GenRand(), r2.GenRand());

        r2 = r1;

        UNIT_ASSERT_VALUES_EQUAL(r1.GenRand(), r2.GenRand());
    }
} // Y_UNIT_TEST_SUITE(TMersenneRndTest)
