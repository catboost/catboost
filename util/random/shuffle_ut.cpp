#include "fast.h"
#include "shuffle.h"
#include "mersenne.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/ylimits.h>

Y_UNIT_TEST_SUITE(TRandUtilsTest) {
    template <typename... A>
    static void TestRange(A&&... args) {
        TString s0, s1;
        ShuffleRange(s1, args...);
        s1 = "0";
        ShuffleRange(s1, args...);
        s1 = "01";
        ShuffleRange(s1, args...);
        s1 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        s0 = s1.copy();
        ShuffleRange(s1, args...);
        UNIT_ASSERT(s0 != s1); // if shuffle does work, chances it will fail are 1 to 64!.
    }

    template <typename... A>
    static void TestIter(A&&... args) {
        TString s0, s1;

        auto f = [&]() {
            auto b = s1.begin();
            auto e = s1.end();

            Shuffle(b, e, args...);
        };

        s1 = "0";
        f();

        s1 = "01";
        f();

        s1 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        s0 = s1.copy();
        f();

        UNIT_ASSERT(s0 != s1); // if shuffle does work, chances it will fail are 1 to 64!.
    }

    template <typename... A>
    static void TestIterPartial(A&&... args) {
        TString s0, s1;

        auto f = [&](int shuffledSize) {
            auto b = s1.begin();
            auto e = s1.end();

            PartialShuffle(b, e, shuffledSize, args...);
        };

        s1 = "";
        f(0);

        s1 = "01";
        f(1);

        s1 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.;?!";
        s0 = s1.copy();
        f(2);
        size_t matchesCount = 0;
        for (size_t i = 0; i < s0.size(); ++i) {
            matchesCount += (s0[i] == s1[i]);
        }

        UNIT_ASSERT(matchesCount >= s1.size() - 4); // partial shuffle doesn't make non-necessary swaps.

        s1 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.;?!";
        s0 = s1.copy();
        f(64);

        UNIT_ASSERT(s0 != s1); // if partial shuffle does work, chances it will fail are 1 to 64!.
    }

    Y_UNIT_TEST(TestShuffle) {
        TestRange();
        TestIterPartial();
    }

    Y_UNIT_TEST(TestShuffleMersenne64) {
        TMersenne<ui64> prng(42);

        TestRange(prng);
        TestIterPartial(prng);
    }

    Y_UNIT_TEST(TestShuffleMersenne32) {
        TMersenne<ui32> prng(24);

        TestIter(prng);
        TestIterPartial(prng);
    }

    Y_UNIT_TEST(TestShuffleFast32) {
        TFastRng32 prng(24, 0);

        TestIter(prng);
        TestIterPartial(prng);
    }

    Y_UNIT_TEST(TestShuffleFast64) {
        TFastRng64 prng(24, 0, 25, 1);

        TestIter(prng);
        TestIterPartial(prng);
    }
} // Y_UNIT_TEST_SUITE(TRandUtilsTest)
