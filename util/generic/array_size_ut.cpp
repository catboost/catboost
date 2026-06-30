#include "array_size.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(ArraySizeTest) {
    Y_UNIT_TEST(Test1) {
        int x[100];
        Y_UNUSED(x); /* Make MSVC happy. */

        UNIT_ASSERT_VALUES_EQUAL(Y_ARRAY_SIZE(x), 100);
    }

    Y_UNIT_TEST(Test2) {
        struct T {
        };

        T x[1];
        Y_UNUSED(x); /* Make MSVC happy. */

        UNIT_ASSERT_VALUES_EQUAL(Y_ARRAY_SIZE(x), 1);
    }
} // Y_UNIT_TEST_SUITE(ArraySizeTest)
