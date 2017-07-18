#include "va_args.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TMacroVarargMapTest) {
    SIMPLE_UNIT_TEST(TestMapArgs) {
        static const char COMBINED[] = Y_MAP_ARGS(Y_STRINGIZE, 1, 2, 3);
        UNIT_ASSERT_STRINGS_EQUAL(COMBINED, "123");
    }

    SIMPLE_UNIT_TEST(TestMapArgsWithLast) {
#define ADD(x) x +
#define ID(x) x
        static const int SUM = Y_MAP_ARGS_WITH_LAST(ADD, ID, 1, 2, 3, 4 + 5);
        UNIT_ASSERT_VALUES_EQUAL(SUM, 1 + 2 + 3 + 4 + 5);
#undef ADD
#undef ID
    }
}
