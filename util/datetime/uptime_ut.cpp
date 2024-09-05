#include <library/cpp/testing/unittest/registar.h>

#include "uptime.h"

Y_UNIT_TEST_SUITE(TestUptimeSuite) {
    Y_UNIT_TEST(TestUptime) {
        auto t1 = Uptime();
        Sleep(TDuration::MilliSeconds(50)); // typical uptime resolution is 10-16 ms
        auto t2 = Uptime();
        UNIT_ASSERT(t2 > t1);
    }
} // Y_UNIT_TEST_SUITE(TestUptimeSuite)
