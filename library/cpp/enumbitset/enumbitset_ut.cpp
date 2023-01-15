#include "enumbitset.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/str.h>
#include <util/stream/file.h>

enum ETestEnum {
    TE_FIRST = 0,
    TE_1,
    TE_2,
    TE_3,
    TE_4,
    TE_MIDDLE = TE_4 + 64,
    TE_5,
    TE_6,
    TE_7,
    TE_MAX,

    TE_OVERFLOW, // to test overflow
};
using TTestBitSet = TEnumBitSet<ETestEnum, TE_FIRST, TE_MAX>;

Y_UNIT_TEST_SUITE(TEnumBitSetTest) {
    Y_UNIT_TEST(TestMainFunctions) {
        auto ebs = TTestBitSet(TE_FIRST, TE_MIDDLE);

        UNIT_ASSERT(ebs.SafeTest(TE_FIRST));
        UNIT_ASSERT(ebs.SafeTest(TE_MIDDLE));

        ebs.SafeSet(TE_5);
        UNIT_ASSERT(ebs.SafeTest(TE_5));
        UNIT_ASSERT(!ebs.SafeTest(TE_7));

        ebs.SafeSet(TE_OVERFLOW);
        UNIT_ASSERT(!ebs.SafeTest(TE_OVERFLOW));
    }

    Y_UNIT_TEST(TestEmpty) {
        TTestBitSet mask;
        UNIT_ASSERT(mask.Empty());
        if (mask)
            UNIT_ASSERT(false && "should be empty");

        mask.Set(TE_FIRST);
        UNIT_ASSERT(!mask.Empty());
        UNIT_ASSERT(mask.Count() == 1);
        if (!mask)
            UNIT_ASSERT(false && "should not be empty");
    }

    Y_UNIT_TEST(TestIter) {
        TTestBitSet mask = TTestBitSet(TE_1, TE_3, TE_7);

        TTestBitSet mask2;
        for (auto elem : mask) {
            mask2.Set(elem);
        }

        UNIT_ASSERT(mask == mask2);
    }

    Y_UNIT_TEST(TestSerialization) {
        auto ebs = TTestBitSet(TE_MIDDLE, TE_6, TE_7);

        TStringStream ss;
        ebs.Save(&ss);

        auto ebs2 = TTestBitSet();
        ebs2.Load(&ss);
        UNIT_ASSERT_EQUAL(ebs, ebs2);
    }

    Y_UNIT_TEST(TestStringRepresentation) {
        auto ebs = TTestBitSet(TE_MIDDLE, TE_6, TE_7);

        UNIT_ASSERT_EQUAL(ebs.ToString(), "D00000000000000000");

        auto ebs2 = TTestBitSet();
        ebs2.FromString("D00000000000000000");
        UNIT_ASSERT_EQUAL(ebs, ebs2);
    }
}
