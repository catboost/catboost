#include "buffer.h"

#include <library/unittest/registar.h>

#include <util/generic/buffer.h>

#include "str.h"

SIMPLE_UNIT_TEST_SUITE(TBufferTest) {
    SIMPLE_UNIT_TEST(Transfer) {
        TBuffer buffer("razrazraz", 9);
        TBufferInput input(buffer);

        input.Skip(3);

        TStringStream output;
        TransferData(&input, &output);

        UNIT_ASSERT_VALUES_EQUAL(output.Str(), "razraz");
    }

    SIMPLE_UNIT_TEST(ReadTo) {
        TBuffer buffer("1234567890", 10);
        TBufferInput input(buffer);

        TString tmp;
        UNIT_ASSERT_VALUES_EQUAL(input.ReadTo(tmp, '3'), 3);
        UNIT_ASSERT_VALUES_EQUAL(tmp, "12");

        UNIT_ASSERT_VALUES_EQUAL(input.ReadTo(tmp, 'z'), 7);
        UNIT_ASSERT_VALUES_EQUAL(tmp, "4567890");
    }
}
