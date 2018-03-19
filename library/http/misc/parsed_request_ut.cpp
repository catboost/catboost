#include "parsed_request.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(THttpParse) {
    SIMPLE_UNIT_TEST(TestParse) {
        TParsedHttpFull h("GET /yandsearch?text=nokia HTTP/1.1");

        UNIT_ASSERT_EQUAL(h.Method, "GET");
        UNIT_ASSERT_EQUAL(h.Request, "/yandsearch?text=nokia");
        UNIT_ASSERT_EQUAL(h.Proto, "HTTP/1.1");

        UNIT_ASSERT_EQUAL(h.Path, "/yandsearch");
        UNIT_ASSERT_EQUAL(h.Cgi, "text=nokia");
    }

    SIMPLE_UNIT_TEST(TestError) {
        bool wasError = false;

        try {
            TParsedHttpFull("GET /yandsearch?text=nokiaHTTP/1.1");
        } catch (...) {
            wasError = true;
        }

        UNIT_ASSERT(wasError);
    }
}
