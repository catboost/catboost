#include "mem.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TestMemIO) {
    SIMPLE_UNIT_TEST(TestReadTo) {
        TString s("0123456789abc");
        TMemoryInput in(s);

        TString t;
        UNIT_ASSERT_VALUES_EQUAL(in.ReadTo(t, '7'), 8);
        UNIT_ASSERT_VALUES_EQUAL(t, "0123456");
        UNIT_ASSERT_VALUES_EQUAL(in.ReadTo(t, 'z'), 5);
        UNIT_ASSERT_VALUES_EQUAL(t, "89abc");
    }

    SIMPLE_UNIT_TEST(Write) {
        char buffer[20];
        TMemoryOutput output(buffer, sizeof(buffer));
        output << "1"
               << "22"
               << "333"
               << "4444"
               << "55555";

        const char* const result = "1"
                                   "22"
                                   "333"
                                   "4444"
                                   "55555";
        UNIT_ASSERT(0 == memcmp(buffer, result, strlen(result)));
    }

    SIMPLE_UNIT_TEST(WriteChars) {
        char buffer[20];
        TMemoryOutput output(buffer, sizeof(buffer));
        output << '1' << '2' << '3' << '4' << '5' << '6' << '7' << '8' << '9' << '0';

        const char* const result = "1234567890";
        UNIT_ASSERT(0 == memcmp(buffer, result, strlen(result)));
    }
}
