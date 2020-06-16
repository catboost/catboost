#include <library/cpp/testing/unittest/registar.h>

#include <library/cpp/getopt/small/wrap.h>

Y_UNIT_TEST_SUITE(Wrap) {
    Y_UNIT_TEST(TestWrapping) {
        UNIT_ASSERT_STRINGS_EQUAL(
            NLastGetopt::Wrap(5, "a b c d eeffeff").Quote(),
            TString("a b c\nd\neeffeff").Quote()
        );

        UNIT_ASSERT_STRINGS_EQUAL(
            NLastGetopt::Wrap(5, "a b\nc d\neeffeff").Quote(),
            TString("a b\nc d\neeffeff").Quote()
        );

        UNIT_ASSERT_STRINGS_EQUAL(
            NLastGetopt::Wrap(5, "a b\n     c d\neeffeff").Quote(),
            TString("a b\n     c\nd\neeffeff").Quote()
        );

        UNIT_ASSERT_STRINGS_EQUAL(
            NLastGetopt::Wrap(5, "a b\nx     c d\neeffeff").Quote(),
            TString("a b\nx\nc d\neeffeff").Quote()
        );

        UNIT_ASSERT_STRINGS_EQUAL(
            NLastGetopt::Wrap(5, "a b\nx  \n   c d\neeffeff").Quote(),
            TString("a b\nx\n   c\nd\neeffeff").Quote()
        );

        UNIT_ASSERT_STRINGS_EQUAL(
            NLastGetopt::Wrap(5, "a b\nx      \n   c d\neeffeff").Quote(),
            TString("a b\nx\n   c\nd\neeffeff").Quote()
        );
    }

    Y_UNIT_TEST(TestWrappingIndent) {
        UNIT_ASSERT_STRINGS_EQUAL(
            NLastGetopt::Wrap(5, "a b c d", "|>").Quote(),
            TString("a b\n|>c d").Quote()
        );

        UNIT_ASSERT_STRINGS_EQUAL(
            NLastGetopt::Wrap(5, "a b\n\nc d", "|>").Quote(),
            TString("a b\n|>\n|>c d").Quote()
        );
    }

    Y_UNIT_TEST(TestWrappingAnsi) {
        UNIT_ASSERT_STRINGS_EQUAL(
            NLastGetopt::Wrap(5, "\033[1;2;3;4;5mx\033[1;2;3;4;5mx\033[1;2;3;4;5mx\033[1;2;3;4;5mx\033[1;2;3;4;5m").Quote(),
            TString("\033[1;2;3;4;5mx\033[1;2;3;4;5mx\033[1;2;3;4;5mx\033[1;2;3;4;5mx\033[1;2;3;4;5m").Quote()
        );

        UNIT_ASSERT_STRINGS_EQUAL(
            NLastGetopt::Wrap(5, "a \033[1;2;3;4;5mb c\033[1;2;3;4;5m \033[1;2;3;4;5md e f").Quote(),
            TString("a \033[1;2;3;4;5mb c\033[1;2;3;4;5m\n\033[1;2;3;4;5md e f").Quote()
        );

        UNIT_ASSERT_STRINGS_EQUAL(
            NLastGetopt::Wrap(5, "a b  \033[1;2;3;4;5m  c d").Quote(),
            TString("a b  \033[1;2;3;4;5m\nc d").Quote()
        );

        UNIT_ASSERT_STRINGS_EQUAL(
            NLastGetopt::Wrap(5, "a b       \033[1;2;3;4;5m  c d").Quote(),
            TString("a b\n\033[1;2;3;4;5m  c d").Quote()
        );
    }

    Y_UNIT_TEST(TestTextInfo) {
        size_t lastLineLen;
        bool hasParagraphs;

        NLastGetopt::Wrap(5, "a b c d e", "", &lastLineLen, &hasParagraphs);
        UNIT_ASSERT_VALUES_EQUAL(lastLineLen, 3);
        UNIT_ASSERT_VALUES_EQUAL(hasParagraphs, false);

        NLastGetopt::Wrap(5, "a b c\n\nd e f h", "", &lastLineLen, &hasParagraphs);
        UNIT_ASSERT_VALUES_EQUAL(lastLineLen, 1);
        UNIT_ASSERT_VALUES_EQUAL(hasParagraphs, true);

        NLastGetopt::Wrap(5, "a b c\n\n", "", &lastLineLen, &hasParagraphs);
        UNIT_ASSERT_VALUES_EQUAL(lastLineLen, 0);
        UNIT_ASSERT_VALUES_EQUAL(hasParagraphs, true);

        NLastGetopt::Wrap(5, "\n \na b c", "", &lastLineLen, &hasParagraphs);
        UNIT_ASSERT_VALUES_EQUAL(lastLineLen, 5);
        UNIT_ASSERT_VALUES_EQUAL(hasParagraphs, true);

        NLastGetopt::Wrap(5, "\nx\na b c", "", &lastLineLen, &hasParagraphs);
        UNIT_ASSERT_VALUES_EQUAL(lastLineLen, 5);
        UNIT_ASSERT_VALUES_EQUAL(hasParagraphs, false);
    }
}
