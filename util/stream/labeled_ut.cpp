#include "str.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TLabeledOutputTest) {
    Y_UNIT_TEST(TBasicTest) {
        TStringStream out;
        int x = 3;
        out << LabeledOutput(x, 1, 2, 3 + 4);
        UNIT_ASSERT_STRINGS_EQUAL(out.Str(), "x = 3, 1 = 1, 2 = 2, 3 + 4 = 7");
    }
} // Y_UNIT_TEST_SUITE(TLabeledOutputTest)
