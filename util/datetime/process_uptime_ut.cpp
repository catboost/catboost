#include <library/cpp/testing/unittest/registar.h>

#include "base.h"
#include "process_uptime.h"
#include "uptime.h"

Y_UNIT_TEST_SUITE(TestProcessUptimeSuite) {
    Y_UNIT_TEST(TestProcessUptime) {
        auto t0 = Uptime();
        auto t1 = ProcessUptime();
        UNIT_ASSERT(t1 < TDuration::Minutes(30));
        UNIT_ASSERT(t0 > t1);
        Sleep(TDuration::MilliSeconds(50)); // typical uptime resolution is 10-16 ms
        auto t2 = ProcessUptime();
        UNIT_ASSERT(t2 >= t1);
    }
} // Y_UNIT_TEST_SUITE(TestProcessUptimeSuite)
