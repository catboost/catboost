#include "compat.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/folder/dirut.h>
#include <util/stream/output.h>

Y_UNIT_TEST_SUITE(TCompatTest) {
    Y_UNIT_TEST(TestGetprogname) {
        getprogname(); // just check it links
    }
} // Y_UNIT_TEST_SUITE(TCompatTest)
