#include "align.h"

#include <library/cpp/testing/unittest/registar.h>

class TAlignTest: public TTestBase {
    UNIT_TEST_SUITE(TAlignTest);
    UNIT_TEST(TestDown)
    UNIT_TEST(TestUp)
    UNIT_TEST_SUITE_END();

private:
    inline void TestDown() {
        UNIT_ASSERT(AlignDown(0, 4) == 0);
        UNIT_ASSERT(AlignDown(1, 4) == 0);
        UNIT_ASSERT(AlignDown(2, 4) == 0);
        UNIT_ASSERT(AlignDown(3, 4) == 0);
        UNIT_ASSERT(AlignDown(4, 4) == 4);
        UNIT_ASSERT(AlignDown(5, 4) == 4);
        UNIT_ASSERT(AlignDown(0, 8) == 0);
        UNIT_ASSERT(AlignDown(1, 8) == 0);
    }

    inline void TestUp() {
        UNIT_ASSERT(AlignUp(0, 4) == 0);
        UNIT_ASSERT(AlignUp(1, 4) == 4);
        UNIT_ASSERT(AlignUp(2, 4) == 4);
        UNIT_ASSERT(AlignUp(3, 4) == 4);
        UNIT_ASSERT(AlignUp(4, 4) == 4);
        UNIT_ASSERT(AlignUp(5, 4) == 8);
        UNIT_ASSERT(AlignUp(0, 8) == 0);
        UNIT_ASSERT(AlignUp(1, 8) == 8);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TAlignTest);
