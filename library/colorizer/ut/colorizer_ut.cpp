#include <library/colorizer/colors.h>

#include <library/unittest/registar.h>

#include <util/string/escape.h>

SIMPLE_UNIT_TEST_SUITE(ColorizerTest) {
    SIMPLE_UNIT_TEST(BasicTest) {
        NColorizer::TColors colors;
        colors.Enable();
        TStringBuf color = colors.BlueColor();
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(color.ToString()), "\\x1B[22;34m");
        colors.Disable();
        UNIT_ASSERT(colors.BlueColor().Empty());
    }

    SIMPLE_UNIT_TEST(ResettingTest) {
        NColorizer::TColors colors;
        colors.Enable();
        // 22;39, not 0, should be used so that only foreground changes
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(colors.OldColor().ToString()), "\\x1B[22;39m");
        // 22, not 0, should be used to reset boldness
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(colors.PurpleColor().ToString()), "\\x1B[22;35m");
    }
}
