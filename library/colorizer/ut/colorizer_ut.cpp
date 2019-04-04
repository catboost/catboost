#include <library/colorizer/colors.h>

#include <library/unittest/registar.h>
#include <util/stream/str.h>

#include <util/string/escape.h>

Y_UNIT_TEST_SUITE(ColorizerTest) {
    Y_UNIT_TEST(BasicTest) {
        NColorizer::TColors colors;
        colors.Enable();
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(colors.BlueColor().ToString()), "\\x1B[22;34m");
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(colors.ForeBlue().ToString()), "\\x1B[34m");
        colors.Disable();
        UNIT_ASSERT(colors.BlueColor().Empty());
    }

    Y_UNIT_TEST(ResettingTest) {
        NColorizer::TColors colors;
        colors.Enable();
        // 22;39, not 0, should be used so that only foreground changes
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(colors.OldColor().ToString()), "\\x1B[22;39m");
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(colors.Reset().ToString()), "\\x1B[0m");
        // 22, not 0, should be used to reset boldness
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(colors.PurpleColor().ToString()), "\\x1B[22;35m");
    }

    Y_UNIT_TEST(PrintAnsi) {
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(ToString(NColorizer::BLUE)), "\\x1B[0m\\x1B[0;34m");
        {
            TString str;
            TStringOutput sink{str};

            sink << NColorizer::BLUE << "foo!" << NColorizer::RESET;

            UNIT_ASSERT_STRINGS_EQUAL(EscapeC(str), "foo!");  // TStringOutput is not tty
        }
        {
            TString str;
            TStringOutput sink{str};

            // Enable this for test purposes. If you're making output of the `AutoColors` constant and this
            // test does not compile, you're free to remove it.
            NColorizer::AutoColors(sink).Enable();

            sink << NColorizer::BLUE << "foo!" << NColorizer::RESET;

            UNIT_ASSERT_STRINGS_EQUAL(EscapeC(str), "\\x1B[0m\\x1B[0;34mfoo!\\x1B[0m");  // TStringOutput is not tty
        }
    }
}
