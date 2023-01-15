#include "bool.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(AtomicBool) {
    Y_UNIT_TEST(ReadWrite) {
        NAtomic::TBool v;

        UNIT_ASSERT_VALUES_EQUAL((bool)v, false);

        v = true;

        UNIT_ASSERT_VALUES_EQUAL((bool)v, true);

        v = false;

        UNIT_ASSERT_VALUES_EQUAL((bool)v, false);

        NAtomic::TBool v2;

        UNIT_ASSERT(v == v2);

        v2 = true;

        UNIT_ASSERT(v != v2);

        v = v2;

        UNIT_ASSERT(v == v2);
    }
}
