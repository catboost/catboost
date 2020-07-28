#include "relaxed_escaper.h"

#include <library/cpp/testing/unittest/registar.h>

#define RESC_FIXED_STR(s) TStringBuf(s, sizeof(s) - 1)
static const TStringBuf CommonTestData[] = {
    // Should be valid UTF-8.
    RESC_FIXED_STR("http://ya.ru/"), RESC_FIXED_STR("http://ya.ru/"),
    RESC_FIXED_STR("http://ya.ru/\\x17\\n"), RESC_FIXED_STR("http://ya.ru/\x17\n"),

    RESC_FIXED_STR("http://ya.ru/\\0"), RESC_FIXED_STR("http://ya.ru/\0"),
    RESC_FIXED_STR("http://ya.ru/\\0\\0"), RESC_FIXED_STR("http://ya.ru/\0\0"),
    RESC_FIXED_STR("http://ya.ru/\\0\\0000"), RESC_FIXED_STR("http://ya.ru/\0\0"
                                                             "0"),
    RESC_FIXED_STR("http://ya.ru/\\0\\0001"), RESC_FIXED_STR("http://ya.ru/\0\x00"
                                                             "1"),

    RESC_FIXED_STR("\\2\\4\\00678"), RESC_FIXED_STR("\2\4\6"
                                                    "78"),
    RESC_FIXED_STR("\\2\\4\\689"), RESC_FIXED_STR("\2\4\689"),

    RESC_FIXED_STR("\\\"Hello\\\", Alice said."), RESC_FIXED_STR("\"Hello\", Alice said."),
    RESC_FIXED_STR("Slash\\\\dash!"), RESC_FIXED_STR("Slash\\dash!"),
    RESC_FIXED_STR("There\\nare\\r\\nnewlines."), RESC_FIXED_STR("There\nare\r\nnewlines."),
    RESC_FIXED_STR("There\\tare\\ttabs."), RESC_FIXED_STR("There\tare\ttabs.")};
#undef RESC_FIXED_STR

Y_UNIT_TEST_SUITE(TRelaxedEscaperTest) {
    Y_UNIT_TEST(TestEscaper) {
        using namespace NEscJ;
        for (size_t i = 0; i < Y_ARRAY_SIZE(CommonTestData); i += 2) {
            TString expected(CommonTestData[i].data(), CommonTestData[i].size());
            TString source(CommonTestData[i + 1].data(), CommonTestData[i + 1].size());
            TString actual(EscapeJ<false>(source));
            TString actual2(UnescapeC(expected));

            UNIT_ASSERT_VALUES_EQUAL(expected, actual);
            UNIT_ASSERT_VALUES_EQUAL(source, actual2);
        }

        UNIT_ASSERT_VALUES_EQUAL("http://ya.ru/\\x17\\n\xAB", EscapeJ<false>("http://ya.ru/\x17\n\xab"));
        TString s = EscapeJ<false, true>("http://ya.ru/\x17\n\xab\xff");
        UNIT_ASSERT_VALUES_EQUAL("http://ya.ru/\\u0017\\n\xAB\\xFF", s);
        UNIT_ASSERT_VALUES_EQUAL("http://ya.ru/\\x17\n\xAB", EscapeJ<false>("http://ya.ru/\x17\n\xab", "\n"));
        UNIT_ASSERT_VALUES_EQUAL("http:\\x2F\\x2Fya.ru\\x2F\\x17\n\xAB'", EscapeJ<false>("http://ya.ru/\x17\n\xab'", "\n'", "/"));
        UNIT_ASSERT_VALUES_EQUAL("http://ya.ru/\x17\n\xab", UnescapeC("http:\\x2F\\x2Fya.ru\\x2F\\x17\n\xAB"));
        UNIT_ASSERT_VALUES_EQUAL("http://ya.ru/\x17\n\xab", UnescapeC("http://ya.ru/\\x17\\n\xAB"));
        UNIT_ASSERT_VALUES_EQUAL("h", EscapeJ<false>("h"));
        UNIT_ASSERT_VALUES_EQUAL("\"h\"", EscapeJ<true>("h"));
        UNIT_ASSERT_VALUES_EQUAL("h", UnescapeC("h"));
        UNIT_ASSERT_VALUES_EQUAL("\\xFF", EscapeJ<false>("\xFF"));
        UNIT_ASSERT_VALUES_EQUAL("\"\\xFF\"", EscapeJ<true>("\xFF"));
        UNIT_ASSERT_VALUES_EQUAL("\xFF", UnescapeC("\\xFF"));

        UNIT_ASSERT_VALUES_EQUAL("\\377f", EscapeJ<false>("\xff"
                                                          "f"));
        UNIT_ASSERT_VALUES_EQUAL("\xff"
                                 "f",
                                 UnescapeC("\\377f"));
        UNIT_ASSERT_VALUES_EQUAL("\\xFFg", EscapeJ<false>("\xff"
                                                          "g"));
        UNIT_ASSERT_VALUES_EQUAL("\xff"
                                 "g",
                                 UnescapeC("\\xFFg"));
    }
}
