#include "demangle.h"

#include <library/cpp/unittest/registar.h>

Y_UNIT_TEST_SUITE(TDemangleTest) {
    Y_UNIT_TEST(SimpleTest) {
        // just check it does not crash or leak
        CppDemangle("hello");
        CppDemangle("");
        CppDemangle("Sfsdf$dfsdfTTSFSDF23234::SDFS:FSDFSDF#$%");
    }
}
