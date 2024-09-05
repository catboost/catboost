#include "buffer.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/buffer.h>

#include <cstring>

#include "str.h"

Y_UNIT_TEST_SUITE(TBufferTest) {
    Y_UNIT_TEST(Transfer) {
        TBuffer buffer("razrazraz", 9);
        TBufferInput input(buffer);

        input.Skip(3);

        TStringStream output;
        TransferData(&input, &output);

        UNIT_ASSERT_VALUES_EQUAL(output.Str(), "razraz");
    }

    Y_UNIT_TEST(ReadTo) {
        TBuffer buffer("1234567890", 10);
        TBufferInput input(buffer);

        TString tmp;
        UNIT_ASSERT_VALUES_EQUAL(input.ReadTo(tmp, '3'), 3);
        UNIT_ASSERT_VALUES_EQUAL(tmp, "12");

        UNIT_ASSERT_VALUES_EQUAL(input.ReadTo(tmp, 'z'), 7);
        UNIT_ASSERT_VALUES_EQUAL(tmp, "4567890");
    }

    Y_UNIT_TEST(WriteViaNextAndUndo) {
        TBuffer buffer;
        TBufferOutput output(buffer);
        TString str;

        for (size_t i = 0; i < 10000; ++i) {
            str.push_back('a' + (i % 20));
        }

        size_t written = 0;
        void* ptr = nullptr;
        while (written < str.size()) {
            size_t bufferSize = output.Next(&ptr);
            UNIT_ASSERT(ptr && bufferSize > 0);
            size_t toWrite = Min(bufferSize, str.size() - written);
            memcpy(ptr, str.begin() + written, toWrite);
            written += toWrite;
            if (toWrite < bufferSize) {
                output.Undo(bufferSize - toWrite);
            }
        }

        UNIT_ASSERT(0 == memcmp(buffer.data(), str.begin(), buffer.size()));
    }

    Y_UNIT_TEST(Write) {
        TBuffer buffer;
        TBufferOutput output(buffer);
        output << "1"
               << "22"
               << "333"
               << "4444"
               << "55555";

        UNIT_ASSERT(0 == memcmp(buffer.data(), "1"
                                               "22"
                                               "333"
                                               "4444"
                                               "55555",
                                buffer.size()));
    }

    Y_UNIT_TEST(WriteChars) {
        TBuffer buffer;
        TBufferOutput output(buffer);
        output << '1' << '2' << '3' << '4' << '5' << '6' << '7' << '8' << '9' << '0';

        UNIT_ASSERT(0 == memcmp(buffer.data(), "1234567890", buffer.size()));
    }
} // Y_UNIT_TEST_SUITE(TBufferTest)
