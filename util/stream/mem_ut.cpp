#include "mem.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TestMemIO) {
    SIMPLE_UNIT_TEST(TestReadTo) {
        TString s("0123456789abc");
        TMemoryInput in(s);

        TString t;
        UNIT_ASSERT_VALUES_EQUAL(in.ReadTo(t, '7'), 8);
        UNIT_ASSERT_VALUES_EQUAL(t, "0123456");
        UNIT_ASSERT_VALUES_EQUAL(in.ReadTo(t, 'z'), 5);
        UNIT_ASSERT_VALUES_EQUAL(t, "89abc");
    }
}
