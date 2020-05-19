#include "codepage.h"
#include "recyr.hh"
#include "wide.h"

#include <library/cpp/unittest/registar.h>

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
    UNIT_TEST(TestUTFFromUnknownPlane);
    UNIT_TEST(TestBrokenMultibyte);
    UNIT_TEST(TestSurrogatePairs);
    UNIT_TEST(TestEncodingHints);
    UNIT_TEST(TestToLower);
    UNIT_TEST(TestToUpper);
    UNIT_TEST(TestUpperLower);
    UNIT_TEST(TestBrokenRune);
    UNIT_TEST(TestCanEncode);
    UNIT_TEST_SUITE_END();

public:
    void TestUTF();
    void TestUTFFromUnknownPlane();
    void TestBrokenMultibyte();
    void TestSurrogatePairs();
    void TestEncodingHints();
    void TestToLower();
    void TestToUpper();

    void TestCanEncode();

    inline void TestUpperLower() {
        const CodePage* cp = CodePageByCharset(CODES_ASCII);
        char tmp[100];

        TStringBuf s = AsStringBuf("abcde");

        TStringBuf upper(tmp, cp->ToUpper(s.begin(), s.end(), tmp));
        UNIT_ASSERT_VALUES_EQUAL(upper, AsStringBuf("ABCDE"));

        TStringBuf lower(tmp, cp->ToLower(upper.begin(), upper.end(), tmp));
        UNIT_ASSERT_VALUES_EQUAL(lower, AsStringBuf("abcde"));
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

void TCodepageTest::TestBrokenMultibyte() {
    const ECharset cp = CODES_EUC_JP;

    const char sampletext[] = {'\xe3'};
    wchar32 recodeResult[100];

    size_t nwritten = 0;
    size_t nread = 0;

    RECODE_RESULT res = RecodeToUnicode(cp, sampletext, recodeResult, Y_ARRAY_SIZE(sampletext), Y_ARRAY_SIZE(recodeResult), nread, nwritten);
    UNIT_ASSERT(res == RECODE_OK);
    UNIT_ASSERT(nread == 1);
    UNIT_ASSERT(nwritten == 0);

    const char bigSample[] = {'\xC3', '\x87', '\xC3', '\x8E', '\xC2', '\xB0', '\xC3', '\x85', '\xC3', '\x85', '\xC3', '\xB8'};
    res = RecodeToUnicode(cp, bigSample, recodeResult, Y_ARRAY_SIZE(bigSample), Y_ARRAY_SIZE(recodeResult), nread, nwritten);
    UNIT_ASSERT(res == RECODE_OK);
    UNIT_ASSERT(nread == Y_ARRAY_SIZE(bigSample));
}

void TCodepageTest::TestUTFFromUnknownPlane() {
    static const wchar32 sampletext[] = {0x61, 0x62, 0x63, 0x20,
                                         0x430, 0x431, 0x432, 0x20,
                                         0x1001, 0x1002, 0x1003, 0x20,
                                         0x10001, 0x10002, 0x10003};

    static const size_t BUFFER_SIZE = 1024;
    char bytebuffer[BUFFER_SIZE];

    size_t readchars = 0;
    size_t writtenbytes = 0;
    size_t samplelen = Y_ARRAY_SIZE(sampletext);

    RECODE_RESULT res = RecodeFromUnicode(CODES_UTF8, sampletext, bytebuffer, samplelen, BUFFER_SIZE, readchars, writtenbytes);

    UNIT_ASSERT(res == RECODE_OK);
    UNIT_ASSERT(samplelen == readchars);

    size_t writtenbytes2 = 0;
    char bytebuffer2[BUFFER_SIZE];
    for (size_t i = 0; i != samplelen; ++i) {
        size_t nwr = 0;
        const int res = RecodeFromUnicode(CODES_UTF8, sampletext[i], bytebuffer2 + writtenbytes2, BUFFER_SIZE - writtenbytes2, nwr);
        UNIT_ASSERT_VALUES_EQUAL(res, int(RECODE_OK));
        writtenbytes2 += nwr;
        UNIT_ASSERT(BUFFER_SIZE > writtenbytes2);
    }
    UNIT_ASSERT_VALUES_EQUAL(TStringBuf(bytebuffer, writtenbytes), TStringBuf(bytebuffer2, writtenbytes2));

    wchar32 charbuffer[BUFFER_SIZE];
    size_t readbytes = 0;
    size_t writtenchars = 0;

    res = RecodeToUnicode(CODES_UNKNOWNPLANE, bytebuffer, charbuffer, writtenbytes, BUFFER_SIZE, readbytes, writtenchars);

    UNIT_ASSERT(res == RECODE_OK);
    UNIT_ASSERT(readbytes == writtenbytes);

    wchar32* charbufferend = charbuffer + writtenchars;
    DecodeUnknownPlane(charbuffer, charbufferend, CODES_UTF8);

    UNIT_ASSERT(charbufferend == charbuffer + samplelen);
    for (size_t i = 0; i < samplelen; ++i)
        UNIT_ASSERT(sampletext[i] == charbuffer[i]);

    // Now, concatenate the thing with an explicit character and retest
    res = RecodeToUnicode(CODES_UNKNOWNPLANE, bytebuffer, charbuffer, writtenbytes, BUFFER_SIZE, readbytes, writtenchars);
    UNIT_ASSERT(res == RECODE_OK);
    UNIT_ASSERT(readbytes == writtenbytes);

    charbuffer[writtenchars] = 0x1234;

    size_t morewrittenchars = 0;
    res = RecodeToUnicode(CODES_UNKNOWNPLANE, bytebuffer, charbuffer + writtenchars + 1, writtenbytes, BUFFER_SIZE, readbytes, morewrittenchars);
    UNIT_ASSERT(res == RECODE_OK);
    UNIT_ASSERT(readbytes == writtenbytes);
    UNIT_ASSERT(writtenchars == morewrittenchars);

    charbuffer[2 * writtenchars + 1] = 0x5678;

    charbufferend = charbuffer + 2 * writtenchars + 2;
    DecodeUnknownPlane(charbuffer, charbufferend, CODES_UTF8);

    UNIT_ASSERT(charbufferend == charbuffer + 2 * samplelen + 2);
    for (size_t i = 0; i < samplelen; ++i) {
        UNIT_ASSERT(sampletext[i] == charbuffer[i]);
        UNIT_ASSERT(sampletext[i] == charbuffer[samplelen + 1 + i]);
    }
    UNIT_ASSERT(0x1234 == charbuffer[samplelen]);
    UNIT_ASSERT(0x5678 == charbuffer[2 * samplelen + 1]);

    // test TChar version
    // bytebuffer of len writtenbytes contains sampletext of len samplelen chars in utf8
    TUtf16String wtr = CharToWide(TStringBuf(bytebuffer, writtenbytes), CODES_UNKNOWNPLANE);
    TChar* strend = wtr.begin() + wtr.size();
    DecodeUnknownPlane(wtr.begin(), strend, CODES_UTF8);
    wtr.resize(strend - wtr.data(), 'Q');
    UNIT_ASSERT_VALUES_EQUAL(wtr.size(), samplelen);
    for (size_t i = 0; i < wtr.size(); ++i) {
        if (sampletext[i] >= 0x10000) {
            UNIT_ASSERT_VALUES_EQUAL(wtr[i], ' ');
        } else {
            UNIT_ASSERT_VALUES_EQUAL(wtr[i], sampletext[i]);
        }
    }
}

static void TestSurrogates(const char* str, const wchar16* wide, size_t wideSize) {
    size_t sSize = strlen(str);
    size_t wSize = sSize * 2;
    TArrayHolder<wchar16> w(new wchar16[wSize]);

    size_t read = 0;
    size_t written = 0;
    RECODE_RESULT res = RecodeToUnicode(CODES_UTF8, str, w.Get(), sSize, wSize, read, written);
    UNIT_ASSERT(res == RECODE_OK);
    UNIT_ASSERT(read == sSize);
    UNIT_ASSERT(written == wideSize);
    UNIT_ASSERT(!memcmp(w.Get(), wide, wideSize));

    TArrayHolder<char> s(new char[sSize]);
    res = RecodeFromUnicode(CODES_UTF8, w.Get(), s.Get(), wideSize, sSize, read, written);
    UNIT_ASSERT(res == RECODE_OK);
    UNIT_ASSERT(read == wideSize);
    UNIT_ASSERT(written == sSize);
    UNIT_ASSERT(!memcmp(s.Get(), str, sSize));
}

void TCodepageTest::TestSurrogatePairs() {
    const char* utf8NonBMP = "\xf4\x80\x89\x84\xf4\x80\x89\x87\xf4\x80\x88\xba";
    wchar16 wNonBMPDummy[] = {0xDBC0, 0xDE44, 0xDBC0, 0xDE47, 0xDBC0, 0xDE3A};
    TestSurrogates(utf8NonBMP, wNonBMPDummy, Y_ARRAY_SIZE(wNonBMPDummy));

    const char* utf8NonBMP2 = "ab\xf4\x80\x89\x87n";
    wchar16 wNonBMPDummy2[] = {'a', 'b', 0xDBC0, 0xDE47, 'n'};
    TestSurrogates(utf8NonBMP2, wNonBMPDummy2, Y_ARRAY_SIZE(wNonBMPDummy2));
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

static void TestCanEncodeEmpty() {
    TWtringBuf empty;
    UNIT_ASSERT(CanBeEncoded(empty, CODES_WIN));
    UNIT_ASSERT(CanBeEncoded(empty, CODES_YANDEX));
    UNIT_ASSERT(CanBeEncoded(empty, CODES_UTF8));
}

static void TestCanEncodeEach(const TWtringBuf& text, ECharset encoding, bool expectedResult) {
    // char by char
    for (size_t i = 0; i < text.size(); ++i) {
        if (CanBeEncoded(text.SubStr(i, 1), encoding) != expectedResult)
            ythrow yexception() << "assertion failed: encoding " << NameByCharset(encoding)
                                << " on '" << text.SubStr(i, 1) << "' (expected " << expectedResult << ")";
    }
    // whole text
    UNIT_ASSERT_EQUAL(CanBeEncoded(text, encoding), expectedResult);
}

void TCodepageTest::TestCanEncode() {
    TestCanEncodeEmpty();

    const TUtf16String lat = u"AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz";
    TestCanEncodeEach(lat, CODES_WIN, true);
    TestCanEncodeEach(lat, CODES_YANDEX, true);
    TestCanEncodeEach(lat, CODES_UTF8, true);

    const TUtf16String rus = u"АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя";
    TestCanEncodeEach(rus, CODES_WIN, true);
    TestCanEncodeEach(rus, CODES_YANDEX, true);
    TestCanEncodeEach(rus, CODES_UTF8, true);

    const TUtf16String ukr = u"ҐґЄєІіЇї";
    TestCanEncodeEach(ukr, CODES_WIN, true);
    TestCanEncodeEach(ukr, CODES_YANDEX, true);
    TestCanEncodeEach(ukr, CODES_UTF8, true);

    const TUtf16String pol = u"ĄĆĘŁŃÓŚŹŻąćęłńóśźż";
    TestCanEncodeEach(pol, CODES_WIN, false);
    TestCanEncodeEach(pol, CODES_YANDEX, true);
    TestCanEncodeEach(pol, CODES_UTF_16BE, true);

    const TUtf16String ger = u"ÄäÖöÜüß";
    TestCanEncodeEach(ger, CODES_WIN, false);
    TestCanEncodeEach(ger, CODES_YANDEX, true);
    TestCanEncodeEach(ger, CODES_UTF_16LE, true);

    const TUtf16String fra1 = u"éàèùâêîôûëïç"; // supported in yandex cp
    const TUtf16String fra2 = u"ÉÀÈÙÂÊÎÔÛËÏŸÿÇ";
    const TUtf16String fra3 = u"ÆæŒœ";
    TestCanEncodeEach(fra1 + fra2 + fra3, CODES_WIN, false);
    TestCanEncodeEach(fra1, CODES_YANDEX, true);
    TestCanEncodeEach(fra2 + fra3, CODES_YANDEX, false);
    TestCanEncodeEach(fra1 + fra2 + fra3, CODES_UTF8, true);

    const TUtf16String kaz = u"ӘәҒғҚқҢңӨөҰұҮүҺһ";
    TestCanEncodeEach(kaz, CODES_WIN, false);
    TestCanEncodeEach(kaz, CODES_YANDEX, false);
    TestCanEncodeEach(kaz, CODES_UTF8, true);
    TestCanEncodeEach(kaz, CODES_KAZWIN, true);

    const TUtf16String tur1 = u"ĞİŞğş";
    const TUtf16String tur = tur1 + u"ı";
    TestCanEncodeEach(tur, CODES_WIN, false);
    TestCanEncodeEach(tur, CODES_YANDEX, false);
    TestCanEncodeEach(tur, CODES_UTF8, true);

    const TUtf16String chi = u"新隶体新隸體";
    TestCanEncodeEach(chi, CODES_WIN, false);
    TestCanEncodeEach(chi, CODES_YANDEX, false);
    TestCanEncodeEach(chi, CODES_UTF8, true);
    TestCanEncodeEach(chi, CODES_UTF_16LE, true);

    const TUtf16String jap = u"漢字仮字交じり文";
    TestCanEncodeEach(jap, CODES_WIN, false);
    TestCanEncodeEach(jap, CODES_YANDEX, false);
    TestCanEncodeEach(jap, CODES_UTF8, true);
    TestCanEncodeEach(jap, CODES_UTF_16BE, true);
}
