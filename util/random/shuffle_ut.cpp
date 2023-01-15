#include "fast.h"
#include "shuffle.h"
#include "mersenne.h"

#include <library/cpp/unittest/registar.h>

#include <util/generic/ylimits.h>

Y_UNIT_TEST_SUITE(TRandUtilsTest) {
    Y_UNIT_TEST(TestShuffle) {
        TString s0, s1;
        ShuffleRange(s1);
        s1 = "0";
        ShuffleRange(s1);
        s1 = "01";
        ShuffleRange(s1);
        s1 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        s0 = s1.copy();
        ShuffleRange(s1);
        UNIT_ASSERT(s0 != s1); // if shuffle does work, chances it will fail are 1 to 64!.
    }
    Y_UNIT_TEST(TestShuffleMersenne64) {
        TMersenne<ui64> prng(42);
        TString s0, s1;
        ShuffleRange(s1, prng);
        s1 = "0";
        ShuffleRange(s1, prng);
        s1 = "01";
        ShuffleRange(s1, prng);
        s1 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        s0 = s1.copy();
        ShuffleRange(s1, prng);
        UNIT_ASSERT(s0 != s1); // if shuffle does work, chances it will fail are 1 to 64!.
    }
    Y_UNIT_TEST(TestShuffleMersenne32) {
        TMersenne<ui32> prng(24);
        TString s0, s1;
        Shuffle(s1.begin(), s1.end(), prng);
        s1 = "0";
        Shuffle(s1.begin(), s1.end(), prng);
        s1 = "01";
        Shuffle(s1.begin(), s1.end(), prng);
        s1 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        s0 = s1.copy();
        Shuffle(s1.begin(), s1.end(), prng);
        UNIT_ASSERT(s0 != s1); // if shuffle does work, chances it will fail are 1 to 64!.
    }
    Y_UNIT_TEST(TestShuffleFast32) {
        TFastRng32 prng(24, 0);
        TString s0, s1;
        Shuffle(s1.begin(), s1.end(), prng);
        s1 = "0";
        Shuffle(s1.begin(), s1.end(), prng);
        s1 = "01";
        Shuffle(s1.begin(), s1.end(), prng);
        s1 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        s0 = s1.copy();
        Shuffle(s1.begin(), s1.end(), prng);
        UNIT_ASSERT(s0 != s1); // if shuffle does work, chances it will fail are 1 to 64!.
    }
    Y_UNIT_TEST(TestShuffleFast64) {
        TFastRng64 prng(24, 0, 25, 1);
        TString s0, s1;
        Shuffle(s1.begin(), s1.end(), prng);
        s1 = "0";
        Shuffle(s1.begin(), s1.end(), prng);
        s1 = "01";
        Shuffle(s1.begin(), s1.end(), prng);
        s1 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        s0 = s1.copy();
        Shuffle(s1.begin(), s1.end(), prng);
        UNIT_ASSERT(s0 != s1); // if shuffle does work, chances it will fail are 1 to 64!.
    }
}
