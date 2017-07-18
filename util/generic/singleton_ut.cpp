#include "singleton.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TestSingleton) {
    struct THuge {
        char Buf[1000000];
        int V = 1234;
    };

    SIMPLE_UNIT_TEST(TestHuge) {
        UNIT_ASSERT_VALUES_EQUAL(*HugeSingleton<int>(), 0);
        UNIT_ASSERT_VALUES_EQUAL(HugeSingleton<THuge>()->V, 1234);
    }
}
