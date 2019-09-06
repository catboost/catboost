#include "mem.h"

#include <library/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestMemIO) {
    Y_UNIT_TEST(TestReadTo) {
        TString s("0123456789abc");
        TMemoryInput in(s);

        TString t;
        UNIT_ASSERT_VALUES_EQUAL(in.ReadTo(t, '7'), 8);
        UNIT_ASSERT_VALUES_EQUAL(t, "0123456");
        UNIT_ASSERT_VALUES_EQUAL(in.ReadTo(t, 'z'), 5);
        UNIT_ASSERT_VALUES_EQUAL(t, "89abc");
    }

    Y_UNIT_TEST(NextAndAdvance) {
        char buffer[20];
        TMemoryOutput output(buffer, sizeof(buffer));
        char* ptr;
        output.Next(&ptr);
        *ptr = '1';
        output.Advance(1);
        output.Next(&ptr);
        *ptr = '2';
        *(ptr + 1) = '2';
        output.Advance(2);
        output.Next(&ptr);
        *ptr = '3';
        *(ptr + 1) = '3';
        *(ptr + 2) = '3';
        output.Advance(3);
        output.Finish();

        const char* const result = "1"
                                   "22"
                                   "333";
        UNIT_ASSERT(0 == memcmp(buffer, result, strlen(result)));
    }

    Y_UNIT_TEST(Write) {
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

    Y_UNIT_TEST(WriteChars) {
        char buffer[20];
        TMemoryOutput output(buffer, sizeof(buffer));
        output << '1' << '2' << '3' << '4' << '5' << '6' << '7' << '8' << '9' << '0';

        const char* const result = "1234567890";
        UNIT_ASSERT(0 == memcmp(buffer, result, strlen(result)));
    }
}
