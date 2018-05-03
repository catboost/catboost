#include "src_location.h"

#include <util/string/builder.h>

#include <library/unittest/registar.h>

static inline TString GenLoc() {
    return TStringBuilder() << __LOCATION__;
}

Y_UNIT_TEST_SUITE(TestLocation) {
    Y_UNIT_TEST(Test1) {
        UNIT_ASSERT_VALUES_EQUAL(GenLoc(), "util/system/src_location_ut.cpp:8");
    }
}
