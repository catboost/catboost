#include "compiler.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TCompilerTest) {
    SIMPLE_UNIT_TEST(TestPragmaNoWshadow) {
        Y_PRAGMA_DIAGNOSTIC_PUSH
        Y_PRAGMA_NO_WSHADOW

        // define two variables with similar names, latest must shadow first
        // and there will be no warning for this

        for (int i = 0; i < 1; ++i) {
            for (int i = 100500; i < 100501; ++i) {
                UNIT_ASSERT_EQUAL(i, 100500);
            }
        }

        Y_PRAGMA_DIAGNOSTIC_POP
    }
}
