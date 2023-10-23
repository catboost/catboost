#include "codepage.h"
#include "wide.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/charset/utf8.h>
#include <util/system/yassert.h>

#if defined(_MSC_VER)
#pragma warning(disable : 4309) /*truncation of constant value*/
#endif

namespace {
    const char yandexUpperCase[] =
        "\x81\x82\x83\x84\x85\x86\x87"
        "\x8E"
        "\xA1\xA2\xA3\xA4\xA5\xA6"
        "\xA8\xA9\xAA\xAB\xAC\xAD\xAE\xAF"
        "\xC0\xC1\xC2\xC3\xC4\xC5\xC6\xC7\xC8\xC9\xCA\xCB\xCC\xCD\xCE\xCF"
        "\xD0\xD1\xD2\xD3\xD4\xD5\xD6\xD7\xD8\xD9\xDA\xDB\xDC\xDD\xDE\xDF";

    const char yandexLowerCase[] =
        "\x91\x92\x93\x94\x95\x96\x97"
        "\x9E"
        "\xB1\xB2\xB3\xB4\xB5\xB6"
        "\xB8\xB9\xBA\xBB\xBC\xBD\xBE\xBF"
        "\xE0\xE1\xE2\xE3\xE4\xE5\xE6\xE7\xE8\xE9\xEA\xEB\xEC\xED\xEE\xEF"
        "\xF0\xF1\xF2\xF3\xF4\xF5\xF6\xF7\xF8\xF9\xFA\xFB\xFC\xFD\xFE\xFF";
}

class TCodepageTest: public TTestBase {
private:
    UNIT_TEST_SUITE(TCodepageTest);
    UNIT_TEST(TestUTF);
    UNIT_TEST(TestEncodingHints);
    UNIT_TEST(TestToLower);
    UNIT_TEST(TestToUpper);
    UNIT_TEST(TestUpperLower);
    UNIT_TEST(TestBrokenRune);
    UNIT_TEST_SUITE_END();

public:
    void TestUTF();
    void TestEncodingHints();
    void TestToLower();
    void TestToUpper();

    inline void TestUpperLower() {
        const CodePage* cp = CodePageByCharset(CODES_ASCII);
        char tmp[100];

        TStringBuf s = "abcde";

        TStringBuf upper(tmp, cp->ToUpper(s.begin(), s.end(), tmp));
        UNIT_ASSERT_VALUES_EQUAL(upper, TStringBuf("ABCDE"));

        TStringBuf lower(tmp, cp->ToLower(upper.begin(), upper.end(), tmp));
        UNIT_ASSERT_VALUES_EQUAL(lower, TStringBuf("abcde"));
    }

    void TestBrokenRune() {
        UNIT_ASSERT_VALUES_EQUAL(BROKEN_RUNE, 0xFFFDu);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TCodepageTest);

void TCodepageTest::TestUTF() {
    for (wchar32 i = 0; i <= 0x10FFFF; i++) {
        unsigned char buffer[32];
        Zero(buffer);
        size_t rune_len;
        size_t ref_len = 0;

        if (i < 0x80)
            ref_len = 1;
        else if (i < 0x800)
            ref_len = 2;
        else if (i < 0x10000)
            ref_len = 3;
        else
            ref_len = 4;

        RECODE_RESULT res = SafeWriteUTF8Char(i, rune_len, buffer, buffer + 32);
        UNIT_ASSERT(res == RECODE_OK);
        UNIT_ASSERT(rune_len == ref_len);

        res = SafeWriteUTF8Char(i, rune_len, buffer, buffer + ref_len - 1);
        UNIT_ASSERT(res == RECODE_EOOUTPUT);

        wchar32 rune;
        res = SafeReadUTF8Char(rune, rune_len, buffer, buffer + 32);
        UNIT_ASSERT(res == RECODE_OK);
        UNIT_ASSERT(rune == i);
        UNIT_ASSERT(rune_len == ref_len);

        res = SafeReadUTF8Char(rune, rune_len, buffer, buffer + ref_len - 1);
        UNIT_ASSERT(res == RECODE_EOINPUT);

        if (ref_len > 1) {
            res = SafeReadUTF8Char(rune, rune_len, buffer + 1, buffer + ref_len);
            UNIT_ASSERT(res == RECODE_BROKENSYMBOL);

            buffer[1] |= 0xC0;
            res = SafeReadUTF8Char(rune, rune_len, buffer, buffer + ref_len);
            UNIT_ASSERT(res == RECODE_BROKENSYMBOL);

            buffer[1] &= 0x3F;
            res = SafeReadUTF8Char(rune, rune_len, buffer, buffer + ref_len);
            UNIT_ASSERT(res == RECODE_BROKENSYMBOL);
        }
    }
    const char* badStrings[] = {
        "\xfe",
        "\xff",
        "\xcc\xc0",
        "\xf4\x90\x80\x80",
        //overlong:
        "\xfe\xfe\xff\xff",
        "\xc0\xaf",
        "\xe0\x80\xaf",
        "\xf0\x80\x80\xaf",
        "\xf8\x80\x80\x80\xaf",
        "\xfc\x80\x80\x80\x80\xaf",
        "\xc1\xbf",
        "\xe0\x9f\xbf",
        "\xf0\x8f\xbf\xbf",
        "\xf8\x87\xbf\xbf\xbf",
        "\xfc\x83\xbf\xbf\xbf\xbf",
        "\xc0\x80",
        "\xe0\x80\x80",
        "\xf0\x80\x80\x80",
        "\xf8\x80\x80\x80\x80",
        "\xfc\x80\x80\x80\x80\x80",
        //UTF-16 surrogate (not covered):
        //"\xed\xa0\x80",
        //"\xed\xad\xbf",
        //"\xed\xae\x80",
        //"\xed\xaf\xbf",
        //"\xed\xb0\x80",
        //"\xed\xbe\x80",
        //"\xed\xbf\xbf",
    };
    for (size_t i = 0; i < Y_ARRAY_SIZE(badStrings); ++i) {
        wchar32 rune;
        const ui8* p = (const ui8*)badStrings[i];
        size_t len;
        RECODE_RESULT res = SafeReadUTF8Char(rune, len, p, p + strlen(badStrings[i]));
        UNIT_ASSERT(res == RECODE_BROKENSYMBOL);
    }
}

void TCodepageTest::TestEncodingHints() {
    UNIT_ASSERT(CODES_WIN == EncodingHintByName("windows-1251"));
    UNIT_ASSERT(CODES_WIN == EncodingHintByName("Windows1251"));
    UNIT_ASSERT(CODES_WIN == EncodingHintByName("WIN1251"));
    UNIT_ASSERT(CODES_WIN == EncodingHintByName("window-cp1251"));
    UNIT_ASSERT(CODES_WIN == EncodingHintByName("!!!CP1251???"));
    UNIT_ASSERT(CODES_WIN == EncodingHintByName("'ansi-cp1251;'"));
    UNIT_ASSERT(CODES_WIN == EncodingHintByName("charset=Microsoft-CP1251;"));

    UNIT_ASSERT(CODES_ISO_EAST == EncodingHintByName("iso-8859-2"));
    UNIT_ASSERT(CODES_ISO_EAST == EncodingHintByName("iso-2"));
    UNIT_ASSERT(CODES_ISO_EAST == EncodingHintByName("iso-latin-2"));
    UNIT_ASSERT(CODES_ISO_EAST == EncodingHintByName("charset=\"Latin2\";"));

    UNIT_ASSERT(CODES_UNKNOWN == EncodingHintByName("widow1251"));
    UNIT_ASSERT(CODES_UNKNOWN == EncodingHintByName("default"));
    UNIT_ASSERT(CODES_UNKNOWN == EncodingHintByName("$phpcharset"));

    UNIT_ASSERT(CODES_UNSUPPORTED != EncodingHintByName("ShiftJIS"));
    UNIT_ASSERT(CODES_UNSUPPORTED != EncodingHintByName("Shift_JIS"));
    UNIT_ASSERT(CODES_UNSUPPORTED != EncodingHintByName("Big5"));
    UNIT_ASSERT(CODES_UNSUPPORTED != EncodingHintByName("euc-kr"));
    UNIT_ASSERT(CODES_UNSUPPORTED != EncodingHintByName("EUC-JP"));
    UNIT_ASSERT(CODES_UNSUPPORTED != EncodingHintByName("charset='Shift_JIS';;"));
    UNIT_ASSERT(CODES_UNSUPPORTED != EncodingHintByName("ISO-2022-KR"));
    UNIT_ASSERT(CODES_UNSUPPORTED != EncodingHintByName("ISO-2022-jp"));
}

void TCodepageTest::TestToLower() {
    TTempBuf buf;
    char* data = buf.Data();
    const size_t n = Y_ARRAY_SIZE(yandexUpperCase); // including NTS
    memcpy(data, yandexUpperCase, n);
    ToLower(data, n - 1);
    UNIT_ASSERT(strcmp(data, yandexLowerCase) == 0);
}

void TCodepageTest::TestToUpper() {
    TTempBuf buf;
    char* data = buf.Data();
    const size_t n = Y_ARRAY_SIZE(yandexLowerCase); // including NTS
    memcpy(data, yandexLowerCase, n);
    ToUpper(data, n - 1);
    UNIT_ASSERT(strcmp(data, yandexUpperCase) == 0);
}
