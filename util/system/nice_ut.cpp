#include "nice.h"

#include "platform.h"

#include <library/cpp/testing/unittest/registar.h>

#ifdef _unix_
    #include <sys/resource.h>

static int GetPriority() {
    return getpriority(PRIO_PROCESS, 0);
}
#endif

Y_UNIT_TEST_SUITE(NiceTest) {
    Y_UNIT_TEST(TestNiceZero) {
        UNIT_ASSERT(Nice(0));
        UNIT_ASSERT(Nice(0));
    }
#ifdef _unix_
    Y_UNIT_TEST(TestNice) {
        int prio = GetPriority();

        if (prio >= 10) {
            return;
        }

        if (Nice(-2)) {
            UNIT_ASSERT_VALUES_EQUAL(GetPriority(), prio - 2);
            prio -= 2;
        } else {
            UNIT_ASSERT_VALUES_EQUAL(GetPriority(), prio);
        }
        UNIT_ASSERT(Nice(1));
        UNIT_ASSERT_VALUES_EQUAL(GetPriority(), prio + 1);
        UNIT_ASSERT(Nice(0));
        UNIT_ASSERT_VALUES_EQUAL(GetPriority(), prio + 1);
        UNIT_ASSERT(Nice(2));
        UNIT_ASSERT_VALUES_EQUAL(GetPriority(), prio + 3);
    }
#endif
} // Y_UNIT_TEST_SUITE(NiceTest)
