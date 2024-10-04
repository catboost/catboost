#include <library/cpp/colorizer/colors.h>

#include <library/cpp/testing/unittest/registar.h>
#include <util/stream/str.h>

#include <util/string/escape.h>

Y_UNIT_TEST_SUITE(ColorizerTest) {
    Y_UNIT_TEST(BasicTest) {
        NColorizer::TColors colors;
        colors.Enable();
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(colors.BlueColor()), "\\x1B[22;34m");
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(colors.ForeBlue()), "\\x1B[34m");
        colors.Disable();
        UNIT_ASSERT(colors.BlueColor().empty());
    }

    Y_UNIT_TEST(ResettingTest) {
        NColorizer::TColors colors;
        colors.Enable();
        // 22;39, not 0, should be used so that only foreground changes
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(colors.OldColor()), "\\x1B[22;39m");
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(colors.Reset()), "\\x1B[0m");
        // 22, not 0, should be used to reset boldness
        UNIT_ASSERT_STRINGS_EQUAL(EscapeC(colors.PurpleColor()), "\\x1B[22;35m");
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

    Y_UNIT_TEST(EscapeCodeLen) {
        UNIT_ASSERT_VALUES_EQUAL(NColorizer::TotalAnsiEscapeCodeLen("some text"), 0);
        UNIT_ASSERT_VALUES_EQUAL(NColorizer::TotalAnsiEscapeCodeLen("some\033[0m text"), 4);
        UNIT_ASSERT_VALUES_EQUAL(NColorizer::TotalAnsiEscapeCodeLen("\033[0msome text"), 4);
        UNIT_ASSERT_VALUES_EQUAL(NColorizer::TotalAnsiEscapeCodeLen("some text\033[0m"), 4);
        UNIT_ASSERT_VALUES_EQUAL(NColorizer::TotalAnsiEscapeCodeLen("some\033[0;1;2;3m text"), 10);
        UNIT_ASSERT_VALUES_EQUAL(NColorizer::TotalAnsiEscapeCodeLen("some\033[0;1;2;3 text"), 0);
        UNIT_ASSERT_VALUES_EQUAL(NColorizer::TotalAnsiEscapeCodeLen("some\0330;1;2;3m text"), 0);
        UNIT_ASSERT_VALUES_EQUAL(NColorizer::TotalAnsiEscapeCodeLen("some [0;1;2;3m text"), 0);
        UNIT_ASSERT_VALUES_EQUAL(NColorizer::TotalAnsiEscapeCodeLen("some\033[m text"), 3);
    }
}
