#include "stack.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TYStackTest) {
    SIMPLE_UNIT_TEST(ExplicitBool) {
        ystack<int> s;

        UNIT_ASSERT(!s);
        UNIT_ASSERT(s.empty());
        s.push(100);
        s.push(200);
        UNIT_ASSERT(s);
        UNIT_ASSERT(!s.empty());
    }
}
