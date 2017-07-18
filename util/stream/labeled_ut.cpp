#include "str.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TLabeledOutputTest) {
    SIMPLE_UNIT_TEST(TBasicTest) {
        TStringStream out;
        int x = 3;
        out << LabeledOutput(x, 1, 2, 3 + 4);
        UNIT_ASSERT_STRINGS_EQUAL(out.Str(), "x = 3, 1 = 1, 2 = 2, 3 + 4 = 7");
    }
}
