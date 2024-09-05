#include "rusage.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TRusageTest) {
    Y_UNIT_TEST(TestRusage) {
        TRusage r;
        // just check it returns something
        r.Fill();
    }
} // Y_UNIT_TEST_SUITE(TRusageTest)
