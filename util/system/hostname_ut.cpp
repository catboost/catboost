#include "hostname.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(THostNameTest){
    SIMPLE_UNIT_TEST(Test1){
        UNIT_ASSERT(*GetHostName() != '?');
}

SIMPLE_UNIT_TEST(TestFQDN) {
    UNIT_ASSERT(*GetFQDNHostName() != '?');
}

SIMPLE_UNIT_TEST(TestIsFQDN) {
    const auto x = GetFQDNHostName();

    try {
        UNIT_ASSERT(IsFQDN(x));
    } catch (...) {
        Cerr << x << Endl;

        throw;
    }
}
}
;
