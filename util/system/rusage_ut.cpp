#include "rusage.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TRusageTest) {
    SIMPLE_UNIT_TEST(TestRusage) {
        TRusage r;
        // just check it returns something
        r.Fill();
    }
}
