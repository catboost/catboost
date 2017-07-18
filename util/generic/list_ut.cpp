#include "list.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TYListSuite) {
    SIMPLE_UNIT_TEST(TestInitializerList) {
        ylist<int> l = {3, 42, 6};
        ylist<int> expected;
        expected.push_back(3);
        expected.push_back(42);
        expected.push_back(6);
        UNIT_ASSERT_VALUES_EQUAL(l, expected);
    }
}
