#include "mem.h"

#include <library/cpp/testing/unittest/registar.h>

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

    Y_UNIT_TEST(NextAndUndo) {
        char buffer[20];
        TMemoryOutput output(buffer, sizeof(buffer));
        char* ptr = nullptr;
        size_t bufferSize = output.Next(&ptr);
        UNIT_ASSERT_GE(bufferSize, 1);
        *ptr = '1';
        if (bufferSize > 1) {
            output.Undo(bufferSize - 1);
        }

        bufferSize = output.Next(&ptr);
        UNIT_ASSERT_GE(bufferSize, 2);
        *ptr = '2';
        *(ptr + 1) = '2';
        if (bufferSize > 2) {
            output.Undo(bufferSize - 2);
        }

        bufferSize = output.Next(&ptr);
        UNIT_ASSERT_GE(bufferSize, 3);
        *ptr = '3';
        *(ptr + 1) = '3';
        *(ptr + 2) = '3';
        if (bufferSize > 3) {
            output.Undo(bufferSize - 3);
        }

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
} // Y_UNIT_TEST_SUITE(TestMemIO)
