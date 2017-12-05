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

    SIMPLE_UNIT_TEST(TestHaveInt128) {
#ifdef Y_HAVE_INT128
        // will be compiled without errors
        unsigned __int128 a = 1;
        __int128 b = 1;
        UNIT_ASSERT_EQUAL(a, 1);
        UNIT_ASSERT_EQUAL(b, 1);
        UNIT_ASSERT_EQUAL(sizeof(a), sizeof(b));

        // and we can set a type alias for __int128 and unsigned __int128 without compiler errors
        using TMyInt128 = __int128;
        using TMyUnsignedInt128 = unsigned __int128;

        TMyInt128 i128value;
        TMyUnsignedInt128 ui128value;
        Y_UNUSED(i128value);
        Y_UNUSED(ui128value);

#endif
    }
}
