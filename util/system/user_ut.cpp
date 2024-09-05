#include "user.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestUser) {
    Y_UNIT_TEST(TestNotEmpty) {
        UNIT_ASSERT(GetUsername());
    }
} // Y_UNIT_TEST_SUITE(TestUser)
