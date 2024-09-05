#include <library/cpp/testing/unittest/registar.h>

#include "cputimer.h"

Y_UNIT_TEST_SUITE(TestCpuTimerSuite) {
    Y_UNIT_TEST(TestCyclesToDurationSafe) {
        ui64 cycles = DurationToCyclesSafe(TDuration::Hours(24));
        UNIT_ASSERT_VALUES_EQUAL(24, CyclesToDurationSafe(cycles).Hours());
    }
} // Y_UNIT_TEST_SUITE(TestCpuTimerSuite)
