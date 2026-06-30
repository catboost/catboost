#include "src_location.h"

#include <util/string/builder.h>

#include <library/cpp/testing/unittest/registar.h>

static inline TString GenLoc() {
    return TStringBuilder() << __LOCATION__;
}

Y_UNIT_TEST_SUITE(TestLocation) {
    Y_UNIT_TEST(Test1) {
        UNIT_ASSERT_VALUES_EQUAL(GenLoc(), "util/system/src_location_ut.cpp:8");

        static constexpr TSourceLocation location = __LOCATION__;
        static_assert(location.Line >= 0, "__LOCATION__ can be used at compile time expressions");
    }
} // Y_UNIT_TEST_SUITE(TestLocation)
