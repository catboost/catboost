#include <library/cpp/unittest/registar.h>

/*
 * just copy-paste it for good start point
 */

Y_UNIT_TEST_SUITE(TUnitTest) {
    Y_UNIT_TEST(TestEqual) {
        UNIT_ASSERT_EQUAL(0, 0);
        UNIT_ASSERT_EQUAL(1, 1);
    }
}
