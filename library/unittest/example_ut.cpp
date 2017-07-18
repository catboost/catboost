#include <library/unittest/registar.h>

/*
 * just copy-paste it for good start point
 */

SIMPLE_UNIT_TEST_SUITE(TUnitTest) {
    SIMPLE_UNIT_TEST(TestEqual) {
        UNIT_ASSERT_EQUAL(0, 0);
        UNIT_ASSERT_EQUAL(1, 1);
    }
}
