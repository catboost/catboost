#include <library/cpp/testing/unittest/registar.h>

#include "httpdate.h"

Y_UNIT_TEST_SUITE(TestHttpDate) {
    Y_UNIT_TEST(Test1) {
        char buf1[100];
        char buf2[100];

        UNIT_ASSERT((int)strlen(format_http_date(0, buf1, sizeof(buf1))) == format_http_date(buf2, sizeof(buf2), 0));
    }
    Y_UNIT_TEST(Test2) {
        UNIT_ASSERT_STRINGS_EQUAL(FormatHttpDate(1234567890), "Fri, 13 Feb 2009 23:31:30 GMT");
    }
}
