#include "wide.h"
#include "codepage.h"
#include "recyr.hh"

#include <library/cpp/testing/unittest/registar.h>

#include <util/charset/utf8.h>
#include <util/digest/numeric.h>
#include <util/generic/hash_set.h>

#include <algorithm>

namespace {
    //! three UTF8 encoded russian letters (A, B, V)
    const char yandexCyrillicAlphabet[] =
        "\xC0\xC1\xC2\xC3\xC4\xC5\xC6\xC7\xC8\xC9\xCA\xCB\xCC\xCD\xCE\xCF"  // A - P
        "\xD0\xD1\xD2\xD3\xD4\xD5\xD6\xD7\xD8\xD9\xDA\xDB\xDC\xDD\xDE\xDF"  // R - YA
        "\xE0\xE1\xE2\xE3\xE4\xE5\xE6\xE7\xE8\xE9\xEA\xEB\xEC\xED\xEE\xEF"  // a - p
        "\xF0\xF1\xF2\xF3\xF4\xF5\xF6\xF7\xF8\xF9\xFA\xFB\xFC\xFD\xFE\xFF"; // r - ya
    const wchar16 wideCyrillicAlphabet[] = {
        0x0410, 0x0411, 0x0412, 0x0413, 0x0414, 0x0415, 0x0416, 0x0417, 0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E, 0x041F,
        0x0420, 0x0421, 0x0422, 0x0423, 0x0424, 0x0425, 0x0426, 0x0427, 0x0428, 0x0429, 0x042A, 0x042B, 0x042C, 0x042D, 0x042E, 0x042F,
        0x0430, 0x0431, 0x0432, 0x0433, 0x0434, 0x0435, 0x0436, 0x0437, 0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E, 0x043F,
        0x0440, 0x0441, 0x0442, 0x0443, 0x0444, 0x0445, 0x0446, 0x0447, 0x0448, 0x0449, 0x044A, 0x044B, 0x044C, 0x044D, 0x044E, 0x044F, 0x00};
    const char utf8CyrillicAlphabet[] =
        "\xd0\x90\xd0\x91\xd0\x92\xd0\x93\xd0\x94\xd0\x95\xd0\x96\xd0\x97"
        "\xd0\x98\xd0\x99\xd0\x9a\xd0\x9b\xd0\x9c\xd0\x9d\xd0\x9e\xd0\x9f"
        "\xd0\xa0\xd0\xa1\xd0\xa2\xd0\xa3\xd0\xa4\xd0\xa5\xd0\xa6\xd0\xa7"
        "\xd0\xa8\xd0\xa9\xd0\xaa\xd0\xab\xd0\xac\xd0\xad\xd0\xae\xd0\xaf"
        "\xd0\xb0\xd0\xb1\xd0\xb2\xd0\xb3\xd0\xb4\xd0\xb5\xd0\xb6\xd0\xb7"
        "\xd0\xb8\xd0\xb9\xd0\xba\xd0\xbb\xd0\xbc\xd0\xbd\xd0\xbe\xd0\xbf"
        "\xd1\x80\xd1\x81\xd1\x82\xd1\x83\xd1\x84\xd1\x85\xd1\x86\xd1\x87"
        "\xd1\x88\xd1\x89\xd1\x8a\xd1\x8b\xd1\x8c\xd1\x8d\xd1\x8e\xd1\x8f";

    TString CreateYandexText() {
        const int len = 256;
        char text[len] = {0};
        for (int i = 0; i < len; ++i) {
            text[i] = static_cast<char>(i);
        }
        return TString(text, len);
    }

    TUtf16String CreateUnicodeText() {
        const int len = 256;
        wchar16 text[len] = {
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, // 0x00 - 0x0F
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, // 0x10 - 0x1F
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, // 0x20 - 0x2F
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, // 0x30 - 0x3F
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, // 0x40 - 0x4F
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, // 0x50 - 0x5F
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, // 0x60 - 0x6F
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, // 0x70 - 0x7F

            0x0301, 0x00C4, 0x00D6, 0x00DC, 0x0104, 0x0106, 0x0118, 0x0141, 0x00E0, 0x00E2, 0x00E7, 0x00E8, 0x00E9, 0x00EA, 0x0490, 0x00AD, // 0x80 - 0x8F
            0x00DF, 0x00E4, 0x00F6, 0x00FC, 0x0105, 0x0107, 0x0119, 0x0142, 0x00EB, 0x00EE, 0x00EF, 0x00F4, 0x00F9, 0x00FB, 0x0491, 0x92CF, // 0x90 - 0x9F
            0x00A0, 0x0143, 0x00D3, 0x015A, 0x017B, 0x0179, 0x046C, 0x00A7, 0x0401, 0x0462, 0x0472, 0x0474, 0x040E, 0x0406, 0x0404, 0x0407, // 0xA0 - 0xAF
            0x00B0, 0x0144, 0x00F3, 0x015B, 0x017C, 0x017A, 0x046D, 0x2116, 0x0451, 0x0463, 0x0473, 0x0475, 0x045E, 0x0456, 0x0454, 0x0457  // 0xB0 - 0xBF
        };
        for (int i = 0; i < len; ++i) {
            if (i <= 0x7F) { // ASCII characters without 0x7 and 0x1B
                text[i] = static_cast<wchar16>(i);
            } else if (i >= 0xC0 && i <= 0xFF) {            // russian characters (without YO and yo)
                text[i] = static_cast<wchar16>(i + 0x0350); // 0x0410 - 0x044F
            }
        }
        return TUtf16String(text, len);
    }

    TString CreateUTF8Text() {
        char text[] = {
            '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', '\x08', '\x09', '\x0a', '\x0b', '\x0c', '\x0d', '\x0e', '\x0f',
            '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a', '\x1b', '\x1c', '\x1d', '\x1e', '\x1f',
            '\x20', '\x21', '\x22', '\x23', '\x24', '\x25', '\x26', '\x27', '\x28', '\x29', '\x2a', '\x2b', '\x2c', '\x2d', '\x2e', '\x2f',
            '\x30', '\x31', '\x32', '\x33', '\x34', '\x35', '\x36', '\x37', '\x38', '\x39', '\x3a', '\x3b', '\x3c', '\x3d', '\x3e', '\x3f',
            '\x40', '\x41', '\x42', '\x43', '\x44', '\x45', '\x46', '\x47', '\x48', '\x49', '\x4a', '\x4b', '\x4c', '\x4d', '\x4e', '\x4f',
            '\x50', '\x51', '\x52', '\x53', '\x54', '\x55', '\x56', '\x57', '\x58', '\x59', '\x5a', '\x5b', '\x5c', '\x5d', '\x5e', '\x5f',
            '\x60', '\x61', '\x62', '\x63', '\x64', '\x65', '\x66', '\x67', '\x68', '\x69', '\x6a', '\x6b', '\x6c', '\x6d', '\x6e', '\x6f',
            '\x70', '\x71', '\x72', '\x73', '\x74', '\x75', '\x76', '\x77', '\x78', '\x79', '\x7a', '\x7b', '\x7c', '\x7d', '\x7e', '\x7f',
            '\xcc', '\x81', '\xc3', '\x84', '\xc3', '\x96', '\xc3', '\x9c', '\xc4', '\x84', '\xc4', '\x86', '\xc4', '\x98', '\xc5', '\x81',
            '\xc3', '\xa0', '\xc3', '\xa2', '\xc3', '\xa7', '\xc3', '\xa8', '\xc3', '\xa9', '\xc3', '\xaa', '\xd2', '\x90', '\xc2', '\xad',
            '\xc3', '\x9f', '\xc3', '\xa4', '\xc3', '\xb6', '\xc3', '\xbc', '\xc4', '\x85', '\xc4', '\x87', '\xc4', '\x99', '\xc5', '\x82',
            '\xc3', '\xab', '\xc3', '\xae', '\xc3', '\xaf', '\xc3', '\xb4', '\xc3', '\xb9', '\xc3', '\xbb', '\xd2', '\x91', '\xe9', '\x8b',
            '\x8f', '\xc2', '\xa0', '\xc5', '\x83', '\xc3', '\x93', '\xc5', '\x9a', '\xc5', '\xbb', '\xc5', '\xb9', '\xd1', '\xac', '\xc2',
            '\xa7', '\xd0', '\x81', '\xd1', '\xa2', '\xd1', '\xb2', '\xd1', '\xb4', '\xd0', '\x8e', '\xd0', '\x86', '\xd0', '\x84', '\xd0',
            '\x87', '\xc2', '\xb0', '\xc5', '\x84', '\xc3', '\xb3', '\xc5', '\x9b', '\xc5', '\xbc', '\xc5', '\xba', '\xd1', '\xad', '\xe2',
            '\x84', '\x96', '\xd1', '\x91', '\xd1', '\xa3', '\xd1', '\xb3', '\xd1', '\xb5', '\xd1', '\x9e', '\xd1', '\x96', '\xd1', '\x94',
            '\xd1', '\x97', '\xd0', '\x90', '\xd0', '\x91', '\xd0', '\x92', '\xd0', '\x93', '\xd0', '\x94', '\xd0', '\x95', '\xd0', '\x96',
            '\xd0', '\x97', '\xd0', '\x98', '\xd0', '\x99', '\xd0', '\x9a', '\xd0', '\x9b', '\xd0', '\x9c', '\xd0', '\x9d', '\xd0', '\x9e',
            '\xd0', '\x9f', '\xd0', '\xa0', '\xd0', '\xa1', '\xd0', '\xa2', '\xd0', '\xa3', '\xd0', '\xa4', '\xd0', '\xa5', '\xd0', '\xa6',
            '\xd0', '\xa7', '\xd0', '\xa8', '\xd0', '\xa9', '\xd0', '\xaa', '\xd0', '\xab', '\xd0', '\xac', '\xd0', '\xad', '\xd0', '\xae',
            '\xd0', '\xaf', '\xd0', '\xb0', '\xd0', '\xb1', '\xd0', '\xb2', '\xd0', '\xb3', '\xd0', '\xb4', '\xd0', '\xb5', '\xd0', '\xb6',
            '\xd0', '\xb7', '\xd0', '\xb8', '\xd0', '\xb9', '\xd0', '\xba', '\xd0', '\xbb', '\xd0', '\xbc', '\xd0', '\xbd', '\xd0', '\xbe',
            '\xd0', '\xbf', '\xd1', '\x80', '\xd1', '\x81', '\xd1', '\x82', '\xd1', '\x83', '\xd1', '\x84', '\xd1', '\x85', '\xd1', '\x86',
            '\xd1', '\x87', '\xd1', '\x88', '\xd1', '\x89', '\xd1', '\x8a', '\xd1', '\x8b', '\xd1', '\x8c', '\xd1', '\x8d', '\xd1', '\x8e',
            '\xd1', '\x8f'};
        return TString(text, Y_ARRAY_SIZE(text));
    }

    //! use this function to dump UTF8 text into a file in case of any changes
    //    void DumpUTF8Text() {
    //        TString s = WideToUTF8(UnicodeText);
    //        std::ofstream f("utf8.txt");
    //        f << std::hex;
    //        for (int i = 0; i < (int)s.size(); ++i) {
    //            f << "0x" << std::setw(2) << std::setfill('0') << (int)(ui8)s[i] << ", ";
    //            if ((i + 1) % 16 == 0)
    //                f << std::endl;
    //        }
    //    }

}

//! this unit tests ensure validity of Yandex-Unicode and UTF8-Unicode conversions
//! @note only those conversions are verified because they are used in index
class TConversionTest: public TTestBase {
private:
    //! @note every of the text can have zeros in the middle
    const TString YandexText;
    const TUtf16String UnicodeText;
    const TString UTF8Text;

private:
    UNIT_TEST_SUITE(TConversionTest);
    UNIT_TEST(TestCharToWide);
    UNIT_TEST(TestWideToChar);
    UNIT_TEST(TestYandexEncoding);
    UNIT_TEST(TestRecodeIntoString);
    UNIT_TEST(TestRecodeAppend);
    UNIT_TEST(TestRecode);
    UNIT_TEST(TestUnicodeLimit);
    UNIT_TEST(TestCanEncode);
    UNIT_TEST_SUITE_END();

public:
    TConversionTest()
        : YandexText(CreateYandexText())
        , UnicodeText(CreateUnicodeText())
        , UTF8Text(CreateUTF8Text())
    {
    }

    void TestCharToWide();
    void TestWideToChar();
    void TestYandexEncoding();
    void TestRecodeIntoString();
    void TestRecodeAppend();
    void TestRecode();
    void TestUnicodeLimit();

    void TestCanEncode();
};

UNIT_TEST_SUITE_REGISTRATION(TConversionTest);

// test conversions (char -> wchar32), (wchar32 -> char) and (wchar32 -> wchar16)
#define TEST_WCHAR32(sbuf, wbuf, enc)                                                                                                 \
    do {                                                                                                                              \
        /* convert char to wchar32 */                                                                                                 \
        TTempBuf tmpbuf1(sbuf.length() * sizeof(wchar32));                                                                            \
        const TBasicStringBuf<wchar32> s4buf = NDetail::NBaseOps::Recode<char>(sbuf, reinterpret_cast<wchar32*>(tmpbuf1.Data()), enc); \
                                                                                                                                      \
        /* convert wchar32 to char */                                                                                                 \
        TTempBuf tmpbuf2(s4buf.length() * 4);                                                                                         \
        const TStringBuf s1buf = NDetail::NBaseOps::Recode(s4buf, tmpbuf2.Data(), enc);                                               \
                                                                                                                                      \
        /* convert wchar32 to wchar16 */                                                                                              \
        const TUtf16String wstr2 = UTF32ToWide(s4buf.data(), s4buf.length());                                                         \
                                                                                                                                      \
        /* test conversions */                                                                                                        \
        UNIT_ASSERT_VALUES_EQUAL(sbuf, s1buf);                                                                                        \
        UNIT_ASSERT_VALUES_EQUAL(wbuf, wstr2);                                                                                        \
    } while (false)

void TConversionTest::TestCharToWide() {
    TUtf16String w = CharToWide(YandexText, CODES_YANDEX);

    UNIT_ASSERT(w.size() == 256);
    UNIT_ASSERT(w.size() == UnicodeText.size());

    for (int i = 0; i < 256; ++i) {
        UNIT_ASSERT_VALUES_EQUAL(w[i], UnicodeText[i]);
    }
}

void TConversionTest::TestWideToChar() {
    TString s = WideToChar(UnicodeText, CODES_YANDEX);

    UNIT_ASSERT(s.size() == 256);
    UNIT_ASSERT(s.size() == YandexText.size());

    for (int i = 0; i < 256; ++i) {
        UNIT_ASSERT_VALUES_EQUAL(s[i], YandexText[i]);
    }
}

static void TestSurrogates(const char* str, const wchar16* wide, size_t wideSize, ECharset enc) {
    TUtf16String w = UTF8ToWide(str);

    UNIT_ASSERT(w.size() == wideSize);
    UNIT_ASSERT(!memcmp(w.c_str(), wide, wideSize));

    TString s = WideToChar(w, enc);

    UNIT_ASSERT(s == str);
}

void TConversionTest::TestYandexEncoding() {
    TUtf16String w = UTF8ToWide(utf8CyrillicAlphabet, strlen(utf8CyrillicAlphabet), csYandex);
    UNIT_ASSERT(w == wideCyrillicAlphabet);
    w = UTF8ToWide(yandexCyrillicAlphabet, strlen(yandexCyrillicAlphabet), csYandex);
    UNIT_ASSERT(w == wideCyrillicAlphabet);

    const char* utf8NonBMP2 = "ab\xf4\x80\x89\x87n";
    wchar16 wNonBMPDummy2[] = {'a', 'b', 0xDBC0, 0xDE47, 'n'};
    TestSurrogates(utf8NonBMP2, wNonBMPDummy2, Y_ARRAY_SIZE(wNonBMPDummy2), CODES_UTF8);

    {
        const char* yandexNonBMP2 = "ab?n";
        UNIT_ASSERT(yandexNonBMP2 == WideToChar(wNonBMPDummy2, Y_ARRAY_SIZE(wNonBMPDummy2), CODES_YANDEX));

        TString temp;
        temp.resize(Y_ARRAY_SIZE(wNonBMPDummy2));
        size_t read = 0;
        size_t written = 0;
        RecodeFromUnicode(CODES_YANDEX, wNonBMPDummy2, temp.begin(), Y_ARRAY_SIZE(wNonBMPDummy2), temp.size(), read, written);
        temp.remove(written);

        UNIT_ASSERT(yandexNonBMP2 == temp);
    }
}

void TConversionTest::TestRecodeIntoString() {
    TString sYandex(UnicodeText.size() * 4, 'x');
    const char* sdata = sYandex.data();
    TStringBuf sres = NDetail::Recode<wchar16>(UnicodeText, sYandex, CODES_YANDEX);
    UNIT_ASSERT(sYandex == YandexText); // same content
    UNIT_ASSERT(sYandex.data() == sdata);     // reserved buffer reused
    UNIT_ASSERT(sYandex.data() == sres.data());     // same buffer
    UNIT_ASSERT(sYandex.size() == sres.size());     // same size
    TEST_WCHAR32(sYandex, UnicodeText, CODES_YANDEX);

    TUtf16String sUnicode;
    sUnicode.reserve(YandexText.size() * 4);
    const wchar16* wdata = sUnicode.data();
    TWtringBuf wres = NDetail::Recode<char>(YandexText, sUnicode, CODES_YANDEX);
    UNIT_ASSERT(sUnicode == UnicodeText); // same content
    UNIT_ASSERT(sUnicode.data() == wdata);      // reserved buffer reused
    UNIT_ASSERT(sUnicode.data() == wres.data());      // same buffer
    UNIT_ASSERT(sUnicode.size() == wres.size());      // same size

    TString sUtf8 = " ";
    size_t scap = sUtf8.capacity();
    sres = NDetail::Recode<wchar16>(UnicodeText, sUtf8, CODES_UTF8);
    UNIT_ASSERT(sUtf8 == UTF8Text);       // same content
    UNIT_ASSERT(sUtf8.capacity() > scap); // increased buffer capacity (supplied was too small)
    UNIT_ASSERT(sUtf8.data() == sres.data());         // same buffer
    UNIT_ASSERT(sUtf8.size() == sres.size());         // same size
    TEST_WCHAR32(sUtf8, UnicodeText, CODES_UTF8);

    sUnicode.clear();
    wdata = sUnicode.data();
    TUtf16String copy = sUnicode; // increase ref-counter
    wres = NDetail::Recode<char>(UTF8Text, sUnicode, CODES_UTF8);
    UNIT_ASSERT(sUnicode == UnicodeText); // same content
#ifndef TSTRING_IS_STD_STRING
    UNIT_ASSERT(sUnicode.data() != wdata);      // re-allocated (shared buffer supplied)
    UNIT_ASSERT(sUnicode.data() == wres.data());      // same buffer
#endif
    UNIT_ASSERT(sUnicode.size() == wres.size());      // same content
}

static TString GenerateJunk(size_t seed) {
    TString res;
    size_t hash = NumericHash(seed);
    size_t size = hash % 1024;
    res.reserve(size);
    for (size_t i = 0; i < size; ++i)
        res += static_cast<char>(NumericHash(hash + i) % 256);
    return res;
}

void TConversionTest::TestRecodeAppend() {
    {
        TString s1, s2;
        NDetail::RecodeAppend<wchar16>(TUtf16String(), s1, CODES_YANDEX);
        UNIT_ASSERT(s1.empty());

        NDetail::RecodeAppend<wchar16>(UnicodeText, s1, CODES_WIN);
        s2 += WideToChar(UnicodeText, CODES_WIN);
        UNIT_ASSERT_EQUAL(s1, s2);

        NDetail::RecodeAppend<wchar16>(UnicodeText, s1, CODES_YANDEX);
        s2 += WideToChar(UnicodeText, CODES_YANDEX);
        UNIT_ASSERT_EQUAL(s1, s2);

        NDetail::RecodeAppend<wchar16>(TUtf16String(), s1, CODES_YANDEX);
        UNIT_ASSERT_EQUAL(s1, s2);

        NDetail::RecodeAppend<wchar16>(UnicodeText, s1, CODES_UTF8);
        s2 += WideToUTF8(UnicodeText);
        UNIT_ASSERT_EQUAL(s1, s2);

        for (size_t i = 0; i < 100; ++i) {
            TUtf16String junk = CharToWide(GenerateJunk(i), CODES_YANDEX);
            NDetail::RecodeAppend<wchar16>(junk, s1, CODES_UTF8);
            s2 += WideToUTF8(junk);
            UNIT_ASSERT_EQUAL(s1, s2);
        }
    }

    {
        TUtf16String s1, s2;
        NDetail::RecodeAppend<char>(TString(), s1, CODES_YANDEX);
        UNIT_ASSERT(s1.empty());

        NDetail::RecodeAppend<char>(YandexText, s1, CODES_WIN);
        s2 += CharToWide(YandexText, CODES_WIN);
        UNIT_ASSERT_EQUAL(s1, s2);

        NDetail::RecodeAppend<char>(YandexText, s1, CODES_YANDEX);
        s2 += CharToWide(YandexText, CODES_YANDEX);
        UNIT_ASSERT_EQUAL(s1, s2);

        NDetail::RecodeAppend<char>(TString(), s1, CODES_YANDEX);
        UNIT_ASSERT_EQUAL(s1, s2);

        NDetail::RecodeAppend<char>(UTF8Text, s1, CODES_UTF8);
        s2 += UTF8ToWide(UTF8Text);
        UNIT_ASSERT_EQUAL(s1, s2);

        for (size_t i = 0; i < 100; ++i) {
            TString junk = GenerateJunk(i);
            NDetail::RecodeAppend<char>(junk, s1, CODES_YANDEX);
            s2 += CharToWide(junk, CODES_YANDEX);
            UNIT_ASSERT_EQUAL(s1, s2);
        }
    }
}

template <>
void Out<RECODE_RESULT>(IOutputStream& out, RECODE_RESULT val) {
    out << int(val);
}

void TConversionTest::TestRecode() {
    for (int c = 0; c != CODES_MAX; ++c) {
        ECharset enc = static_cast<ECharset>(c);
        if (!SingleByteCodepage(enc))
            continue;

        using THash = THashSet<char>;
        THash hash;

        for (int i = 0; i != 256; ++i) {
            char ch = static_cast<char>(i);

            wchar32 wch;
            size_t read = 0;
            size_t written = 0;
            RECODE_RESULT res = RECODE_ERROR;

            res = RecodeToUnicode(enc, &ch, &wch, 1, 1, read, written);
            UNIT_ASSERT(res == RECODE_OK);
            if (wch == BROKEN_RUNE)
                continue;

            char rch = 0;
            res = RecodeFromUnicode(enc, &wch, &rch, 1, 1, read, written);
            UNIT_ASSERT(res == RECODE_OK);

            char rch2 = 0;
            UNIT_ASSERT_VALUES_EQUAL(RECODE_OK, RecodeFromUnicode(enc, wch, &rch2, 1, written));
            UNIT_ASSERT_VALUES_EQUAL(size_t(1), written);
            UNIT_ASSERT_VALUES_EQUAL(rch2, rch);

            if (hash.contains(rch)) { // there are some stupid encodings with duplicate characters
                continue;
            } else {
                hash.insert(rch);
            }

            UNIT_ASSERT(ch == rch);
        }
    }
}

void TConversionTest::TestUnicodeLimit() {
    for (int i = 0; i != CODES_MAX; ++i) {
        ECharset code = static_cast<ECharset>(i);
        if (!SingleByteCodepage(code))
            continue;

        const CodePage* page = CodePageByCharset(code);
        Y_ASSERT(page);

        for (int c = 0; c < 256; ++c) {
            UNIT_ASSERT(page->unicode[c] < 1 << 16);
        }
    }
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

void TConversionTest::TestCanEncode() {
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
