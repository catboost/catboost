#include "demangle.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TDemangleTest) {
    SIMPLE_UNIT_TEST(SimpleTest) {
        // just check it does not crash or leak
        CppDemangle("hello");
        CppDemangle("");
        CppDemangle("Sfsdf$dfsdfTTSFSDF23234::SDFS:FSDFSDF#$%");
    }
}
