#include "stack.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TYStackTest) {
    Y_UNIT_TEST(ExplicitBool) {
        TStack<int> s;

        UNIT_ASSERT(!s);
        UNIT_ASSERT(s.empty());
        s.push(100);
        s.push(200);
        UNIT_ASSERT(s);
        UNIT_ASSERT(!s.empty());
    }
} // Y_UNIT_TEST_SUITE(TYStackTest)
