#include "compat.h"

#include <library/unittest/registar.h>

#include <util/folder/dirut.h>
#include <util/stream/output.h>

SIMPLE_UNIT_TEST_SUITE(TCompatTest) {
    SIMPLE_UNIT_TEST(TestGetprogname) {
        getprogname(); // just check it links
    }
}
