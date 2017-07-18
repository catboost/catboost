#include "user.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TestUser) {
    SIMPLE_UNIT_TEST(TestNotEmpty) {
        UNIT_ASSERT(GetUsername());
    }
}
