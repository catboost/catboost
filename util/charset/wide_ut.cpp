#include "utf8.h"
#include "wide.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/string/reverse.h>

#include <algorithm>

namespace {
    //! three UTF8 encoded russian letters (A, B, V)
    const char utext[] = "\xd0\x90\xd0\x91\xd0\x92";

    const char asciiLatinAlphabet[] = "ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz";
    const wchar16 wideLatinAlphabet[] = {
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'G', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'g', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 0};
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

    const wchar32 LEAD_BITS_MASK_2_BYTES = 0x1F;
    const wchar32 LEAD_BITS_MASK_3_BYTES = 0x0F;
    const wchar32 LEAD_BITS_MASK_4_BYTES = 0x07;

    wchar16 ws[] = {
        0x0009,
        0x000A, 0x2028, 0x2029,
        0x000B,
        0x000C,
        0x000D,
        0x0020, 0x1680,
        0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x200B,
        0x202F, 0x205F, 0x3000,
        0x00A0};

    const size_t CaseTestDataSize = 10;
    wchar32 WideStringTestData[][CaseTestDataSize] = {
        {0x01C4, 0x10428, 0x10429, 0x10447, 0x10441, 0x1C03, 0x00A0, 0x10400, 0x10415, 0x10437}, // original
        {0x01C6, 0x10428, 0x10429, 0x10447, 0x10441, 0x1C03, 0x00A0, 0x10428, 0x1043D, 0x10437}, // lower
        {0x01C4, 0x10400, 0x10401, 0x1041F, 0x10419, 0x1C03, 0x00A0, 0x10400, 0x10415, 0x1040F}, // upper
        {0x01C5, 0x10428, 0x10429, 0x10447, 0x10441, 0x1C03, 0x00A0, 0x10428, 0x1043D, 0x10437}, // title
    };

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
    // void DumpUTF8Text() {
    //     TString s = WideToUTF8(UnicodeText);
    //     std::ofstream f("utf8.txt");
    //     f << std::hex;
    //     for (int i = 0; i < (int)s.size(); ++i) {
    //         f << "0x" << std::setw(2) << std::setfill('0') << (int)(ui8)s[i] << ", ";
    //         if ((i + 1) % 16 == 0)
    //             f << std::endl;
    //     }
    // }

    template <StrictUTF8 strictMode = StrictUTF8::No>
    void CheckRecodeOK(wchar32 expected, unsigned char* first, size_t n) {
        wchar32 w = 0;
        const unsigned char* p = first;

        RECODE_RESULT r = ReadUTF8CharAndAdvance<strictMode>(w, p, first + n);
        UNIT_ASSERT(w == expected);
        UNIT_ASSERT(size_t(p - first) == n);
        UNIT_ASSERT(r == RECODE_OK);
    }

    template <StrictUTF8 strictMode = StrictUTF8::No>
    void CheckBrokenSymbol(unsigned char* first, unsigned char* last) {
        wchar32 w = 0;
        const unsigned char* p = first;

        RECODE_RESULT r = ReadUTF8CharAndAdvance<strictMode>(w, p, last);
        UNIT_ASSERT(w == BROKEN_RUNE);
        UNIT_ASSERT(p - first == 0);
        UNIT_ASSERT(r == RECODE_BROKENSYMBOL);
    }

    void CheckEndOfInput(unsigned char* first, size_t n) {
        wchar32 w = 0;
        const unsigned char* p = first;

        RECODE_RESULT r = ReadUTF8CharAndAdvance(w, p, first + n);
        (void)w;
        UNIT_ASSERT(p - first == 0);
        UNIT_ASSERT(r == RECODE_EOINPUT);
    }

    void CheckCharLen(unsigned char* first, unsigned char* last, size_t len, RECODE_RESULT result) {
        size_t n = 0;
        RECODE_RESULT r = GetUTF8CharLen(n, first, last);
        UNIT_ASSERT(n == len);
        UNIT_ASSERT(r == result);
    }
}

class TConversionTest: public TTestBase {
private:
    //! @note every of the text can have zeros in the middle
    const TUtf16String UnicodeText_;
    const TString Utf8Text_;

private:
    UNIT_TEST_SUITE(TConversionTest);
    UNIT_TEST(TestReadUTF8Char);
    UNIT_TEST(TestGetUTF8CharLen);
    UNIT_TEST(TestWriteUTF8Char);
    UNIT_TEST(TestUTF8ToWide);
    UNIT_TEST(TestWideToUTF8);
    UNIT_TEST(TestGetNumOfUTF8Chars);
    UNIT_TEST(TestSubstrUTF8);
    UNIT_TEST(TestUnicodeCase);
    UNIT_TEST(TestUnicodeDetails);
    UNIT_TEST(TestHexConversion);
    UNIT_TEST_SUITE_END();

public:
    TConversionTest()
        : UnicodeText_(CreateUnicodeText())
        , Utf8Text_(CreateUTF8Text())
    {
    }

    void TestReadUTF8Char();
    void TestGetUTF8CharLen();
    void TestWriteUTF8Char();
    void TestUTF8ToWide();
    void TestWideToUTF8();
    void TestGetNumOfUTF8Chars();
    void TestSubstrUTF8();
    void TestUnicodeCase();
    void TestUnicodeDetails();
    void TestHexConversion();
};

UNIT_TEST_SUITE_REGISTRATION(TConversionTest);

void TConversionTest::TestHexConversion() {
    for (char ch = '0'; ch <= '9'; ++ch) {
        UNIT_ASSERT(isxdigit(ch));
        UNIT_ASSERT(IsHexdigit(ch));
    }

    for (char ch = 'a'; ch <= 'f'; ++ch) {
        UNIT_ASSERT(isxdigit(ch));
        UNIT_ASSERT(IsHexdigit(ch));
    }

    for (char ch = 'A'; ch <= 'F'; ++ch) {
        UNIT_ASSERT(isxdigit(ch));
        UNIT_ASSERT(IsHexdigit(ch));
    }

    for (wchar16 i = std::numeric_limits<wchar16>::min(); i < std::numeric_limits<wchar16>::max(); ++i) {
        if (IsHexdigit(i)) {
            UNIT_ASSERT(isxdigit(char(i)));
        }
    }
}

void TConversionTest::TestReadUTF8Char() {
    wchar32 e; // expected unicode char
    wchar32 c;
    unsigned long u; // single UTF8 encoded character
    unsigned char* const first = reinterpret_cast<unsigned char*>(&u);
    unsigned char* const last = first + sizeof(u);

    // all ASCII characters are converted with no change (zero converted successfully as well)
    for (c = 0; c <= 0x7F; ++c) {
        u = c;
        CheckRecodeOK(c, first, 1);
    }

    // broken symbols from the second half of ASCII table (1000 0000 - 1011 1111)
    for (c = 0x80; c <= 0xBF; ++c) {
        u = c;
        CheckBrokenSymbol(first, last);
    }

    // overlong encoding: leading byte of 2-byte symbol: 1100 0000 - 1100 0001
    for (c = 0xC0; c <= 0xC1; ++c) {
        u = c;
        CheckBrokenSymbol(first, last);

        u |= 0x8000;
        CheckBrokenSymbol(first, first + 2);

        CheckEndOfInput(first, 1);
    }

    // leading byte of 2-byte symbol: 1100 0000 - 1101 1111
    for (c = 0xC2; c <= 0xDF; ++c) {
        u = c;
        CheckBrokenSymbol(first, last);

        u |= 0x8000;
        // w: 0000 0000  0000 0000 - 0000 0111  1100 0000
        e = c & LEAD_BITS_MASK_2_BYTES;
        e <<= 6;
        CheckRecodeOK(e, first, 2);

        CheckEndOfInput(first, 1);
    }

    // possible overlong encoding with leading byte 1110 0000
    {
        u = c = 0xE0;
        CheckBrokenSymbol(first, last);

        u |= 0x808000;
        CheckBrokenSymbol(first, first + 3);

        u = c | 0x80A000;
        e = 0x800;
        CheckRecodeOK(e, first, 3);

        CheckEndOfInput(first, 2);
        CheckEndOfInput(first, 1);
    }

    // leading byte of 3-byte symbol: 1110 0001 - 1110 1111
    for (c = 0xE1; c <= 0xEF; ++c) {
        u = c;
        CheckBrokenSymbol(first, last);

        u |= 0x808000;
        // w: 0000 0000  0000 0000 - 0000 0111  1100 0000
        e = c & LEAD_BITS_MASK_3_BYTES;
        e <<= 12;
        CheckRecodeOK(e, first, 3);

        CheckEndOfInput(first, 2);
        CheckEndOfInput(first, 1);
    }

    // leading byte of 3-byte symbol before surrogates: 1110 0001 - 1110 1100
    for (c = 0xE1; c <= 0xEC; ++c) {
        u = c;
        CheckBrokenSymbol<StrictUTF8::Yes>(first, last);

        u |= 0x808000;
        // w: 0000 0000  0000 0000 - 0000 0111  1100 0000
        e = c & LEAD_BITS_MASK_3_BYTES;
        e <<= 12;
        CheckRecodeOK<StrictUTF8::Yes>(e, first, 3);

        CheckEndOfInput(first, 2);
        CheckEndOfInput(first, 1);
    }

    // rest of allowed characters before surrogate block
    {
        u = 0xED;
        CheckBrokenSymbol<StrictUTF8::Yes>(first, last);

        u |= 0xBF9F00;
        e = 0xD7FF;
        CheckRecodeOK<StrictUTF8::Yes>(e, first, 3);

        CheckEndOfInput(first, 2);
        CheckEndOfInput(first, 1);
    }

    // rfc3629 section 4 forbids characters 0xD800 - 0xDFFF
    {
        u = 0xED;
        CheckBrokenSymbol<StrictUTF8::Yes>(first, last);

        u |= 0x80A000;
        CheckBrokenSymbol<StrictUTF8::Yes>(first, last);

        CheckEndOfInput(first, 2);
        CheckEndOfInput(first, 1);
    }

    // leading byte of 3-byte symbol after surrogates: 1110 1110 - 1110 1111
    for (c = 0xEE; c <= 0xEF; ++c) {
        u = c;
        CheckBrokenSymbol<StrictUTF8::Yes>(first, last);

        u |= 0x808000;
        // w: 0000 0000  0000 0000 - 0000 0111  1100 0000
        e = c & LEAD_BITS_MASK_3_BYTES;
        e <<= 12;
        CheckRecodeOK<StrictUTF8::Yes>(e, first, 3);

        CheckEndOfInput(first, 2);
        CheckEndOfInput(first, 1);
    }

    // possible overlong encoding with leading byte 1111 0000
    {
        u = c = 0xF0;
        CheckBrokenSymbol(first, last);

        u |= 0x80808000;
        CheckBrokenSymbol(first, first + 4);

        u = c | 0x80809000;
        e = 0x10000;
        CheckRecodeOK(e, first, 4);

        CheckEndOfInput(first, 3);
        CheckEndOfInput(first, 2);
        CheckEndOfInput(first, 1);
    }

    // leading byte of 4-byte symbol: 1111 0001 - 1111 0111
    for (c = 0xF1; c <= 0xF3; ++c) {
        u = c;
        CheckBrokenSymbol(first, last);

        u |= 0x80808000;
        // w: 0000 0000  0000 0000 - 0000 0111  1100 0000
        e = c & LEAD_BITS_MASK_4_BYTES;
        e <<= 18;
        CheckRecodeOK(e, first, 4);

        CheckEndOfInput(first, 3);
        CheckEndOfInput(first, 2);
        CheckEndOfInput(first, 1);
    }

    // possible invalid code points with leading byte 1111 0100
    {
        c = 0xF4;

        u = 0x80808000 | c;
        e = c & LEAD_BITS_MASK_4_BYTES;
        e <<= 18;
        CheckRecodeOK(e, first, 4);

        // the largest possible Unicode code point
        u = 0xBFBF8F00 | c;
        e = 0x10FFFF;
        CheckRecodeOK(e, first, 4);

        u = 0x80809000 | c;
        CheckBrokenSymbol(first, last);
    }

    // broken symbols: 1111 0101 - 1111 1111
    for (c = 0xF5; c <= 0xFF; ++c) {
        u = c;
        CheckBrokenSymbol(first, last);
    }
}

void TConversionTest::TestGetUTF8CharLen() {
    wchar32 c;
    unsigned long u; // single UTF8 encoded character
    unsigned char* const first = reinterpret_cast<unsigned char*>(&u);
    unsigned char* const last = first + sizeof(u);

    // all ASCII characters are converted with no change (zero converted successfully as well)
    for (c = 0; c <= 0x7F; ++c) {
        u = c;
        CheckCharLen(first, last, 1, RECODE_OK);
    }

    // broken symbols from the second half of ASCII table (1000 0000 - 1011 1111)
    for (c = 0x80; c <= 0xBF; ++c) {
        u = c;
        CheckCharLen(first, last, 0, RECODE_BROKENSYMBOL);
    }

    // leading byte of 2-byte symbol: 1100 0000 - 1101 1111
    for (c = 0xC0; c <= 0xDF; ++c) {
        u = c;
        CheckCharLen(first, last, 0, RECODE_BROKENSYMBOL);

        u |= 0x8000;
        // w: 0000 0000  0000 0000 - 0000 0111  1100 0000
        CheckCharLen(first, last, 2, RECODE_OK);

        CheckCharLen(first, first + 1, 0, RECODE_EOINPUT);
    }

    // leading byte of 3-byte symbol: 1110 0000 - 1110 1111
    for (c = 0xE0; c <= 0xEF; ++c) {
        u = c;
        CheckCharLen(first, last, 0, RECODE_BROKENSYMBOL);

        u |= 0x808000;
        // w: 0000 0000  0000 0000 - 0000 0111  1100 0000
        CheckCharLen(first, last, 3, RECODE_OK);

        CheckCharLen(first, first + 2, 0, RECODE_EOINPUT);
        CheckCharLen(first, first + 1, 0, RECODE_EOINPUT);
    }

    // leading byte of 4-byte symbol: 1111 0000 - 1111 0111
    for (c = 0xF0; c <= 0xF3; ++c) {
        u = c;
        CheckCharLen(first, last, 0, RECODE_BROKENSYMBOL);

        u |= 0x80808000;
        // w: 0000 0000  0000 0000 - 0000 0111  1100 0000
        CheckCharLen(first, last, 4, RECODE_OK);

        CheckCharLen(first, first + 3, 0, RECODE_EOINPUT);
        CheckCharLen(first, first + 2, 0, RECODE_EOINPUT);
        CheckCharLen(first, first + 1, 0, RECODE_EOINPUT);
    }

    // broken symbols: 1111 1000 - 1111 1111
    for (c = 0xF8; c <= 0xFF; ++c) {
        u = c;
        CheckCharLen(first, last, 0, RECODE_BROKENSYMBOL);
    }
}

void TConversionTest::TestWriteUTF8Char() {
    wchar32 w;
    unsigned long u; // single UTF8 encoded character
    size_t n;

    for (w = 0x00; w < 0x80; ++w) {
        u = 0;
        WriteUTF8Char(w, n, reinterpret_cast<unsigned char*>(&u));
        UNIT_ASSERT((u & 0xFFFFFF80) == 0x00000000);
        UNIT_ASSERT(n == 1);
    }

    for (w = 0x80; w < 0x800; ++w) {
        u = 0;
        WriteUTF8Char(w, n, reinterpret_cast<unsigned char*>(&u));
        UNIT_ASSERT((u & 0xFFFFC000) == 0x00008000); // see constants in ReadUTF8Char
        UNIT_ASSERT(n == 2);
    }

    for (w = 0x800; w < 0x10000; ++w) {
        u = 0;
        WriteUTF8Char(w, n, reinterpret_cast<unsigned char*>(&u));
        UNIT_ASSERT((u & 0xFFC0C000) == 0x00808000); // see constants in ReadUTF8Char
        UNIT_ASSERT(n == 3);
    }

    for (w = 0x10000; w < 0x80; ++w) {
        WriteUTF8Char(w, n, reinterpret_cast<unsigned char*>(&u));
        UNIT_ASSERT((u & 0xC0C0C000) == 0x80808000); // see constants in ReadUTF8Char
        UNIT_ASSERT(n == 4);
    }
}

static void TestSurrogates(const char* str, const wchar16* wide, size_t wideSize) {
    TUtf16String w = UTF8ToWide(str);

    UNIT_ASSERT(w.size() == wideSize);
    UNIT_ASSERT(!memcmp(w.c_str(), wide, wideSize));

    TString s = WideToUTF8(w);

    UNIT_ASSERT(s == str);
}

void TConversionTest::TestUTF8ToWide() {
    TUtf16String w = UTF8ToWide(Utf8Text_);

    UNIT_ASSERT(w.size() == 256);
    UNIT_ASSERT(w.size() == UnicodeText_.size());

    for (int i = 0; i < 256; ++i) {
        UNIT_ASSERT_VALUES_EQUAL(w[i], UnicodeText_[i]);
    }

    wchar16 buffer[4] = {0};
    size_t written = 0;
    // the function must extract 2 symbols only
    bool result = UTF8ToWide(utext, 5, buffer, written);
    UNIT_ASSERT(!result);
    UNIT_ASSERT(buffer[0] == 0x0410);
    UNIT_ASSERT(buffer[1] == 0x0411);
    UNIT_ASSERT(buffer[2] == 0x0000);
    UNIT_ASSERT(buffer[3] == 0x0000);
    UNIT_ASSERT(written == 2);

    memset(buffer, 0, 4);
    written = 0;
    result = UTF8ToWide(utext, 1, buffer, written);
    UNIT_ASSERT(!result);
    UNIT_ASSERT(buffer[0] == 0x0000);
    UNIT_ASSERT(buffer[1] == 0x0000);
    UNIT_ASSERT(buffer[2] == 0x0000);
    UNIT_ASSERT(buffer[3] == 0x0000);
    UNIT_ASSERT(written == 0);

    w = UTF8ToWide(asciiLatinAlphabet, strlen(asciiLatinAlphabet));
    UNIT_ASSERT(w == wideLatinAlphabet);
    w = UTF8ToWide(utf8CyrillicAlphabet, strlen(utf8CyrillicAlphabet));
    UNIT_ASSERT(w == wideCyrillicAlphabet);

    const char* utf8NonBMP = "\xf4\x80\x89\x84\xf4\x80\x89\x87\xf4\x80\x88\xba";
    wchar16 wNonBMPDummy[] = {0xDBC0, 0xDE44, 0xDBC0, 0xDE47, 0xDBC0, 0xDE3A};
    TestSurrogates(utf8NonBMP, wNonBMPDummy, Y_ARRAY_SIZE(wNonBMPDummy));

    const char* utf8NonBMP2 = "ab\xf4\x80\x89\x87n";
    wchar16 wNonBMPDummy2[] = {'a', 'b', 0xDBC0, 0xDE47, 'n'};
    TestSurrogates(utf8NonBMP2, wNonBMPDummy2, Y_ARRAY_SIZE(wNonBMPDummy2));

    UNIT_ASSERT_VALUES_EQUAL(WideToUTF8(UTF8ToWide(WideToUTF8(UTF8ToWide<true>(
                                 "m\xFB\xB2\xA5\xAA\xAFyeuse.sexwebcamz.com")))),
                             TString(
                                 "m\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBDyeuse.sexwebcamz.com"));
}

void TConversionTest::TestWideToUTF8() {
    TString s = WideToUTF8(UnicodeText_);
    size_t len = 0;
    for (TUtf16String::const_iterator i = UnicodeText_.begin(), ie = UnicodeText_.end(); i != ie; ++i) {
        len += UTF8RuneLenByUCS(*i);
    }

    UNIT_ASSERT(s.size() == Utf8Text_.size());
    UNIT_ASSERT(s.size() == len);

    for (int i = 0; i < static_cast<int>(s.size()); ++i) {
        UNIT_ASSERT_VALUES_EQUAL(s[i], Utf8Text_[i]);
    }
}

void TConversionTest::TestGetNumOfUTF8Chars() {
    size_t n = 0;
    bool result = GetNumberOfUTF8Chars(Utf8Text_.c_str(), Utf8Text_.size(), n);
    UNIT_ASSERT(result);
    UNIT_ASSERT(n == 256);

    n = 0;
    result = GetNumberOfUTF8Chars(utext, 5, n);
    UNIT_ASSERT(!result);
    UNIT_ASSERT(n == 2);

    n = 0;
    result = GetNumberOfUTF8Chars(utext, 1, n);
    UNIT_ASSERT(!result);
    UNIT_ASSERT(n == 0);

    UNIT_ASSERT_EQUAL(GetNumberOfUTF8Chars("привет!"), 7);
}

void TConversionTest::TestSubstrUTF8() {
    TStringBuf utextBuf(utext, sizeof(utext));
    UNIT_ASSERT(SubstrUTF8(utextBuf, 0, 2) == utextBuf.substr(0, 4));
    UNIT_ASSERT(SubstrUTF8(utextBuf, 1, 1) == utextBuf.substr(2, 2));
    UNIT_ASSERT(SubstrUTF8(utextBuf, 1, 2) == utextBuf.substr(2, 4));
    UNIT_ASSERT(SubstrUTF8(utextBuf, 1, 3) == utextBuf.substr(2, 6));
}

inline bool MustBeSurrogate(wchar32 ch) {
    return ch > 0xFFFF;
}

void TConversionTest::TestUnicodeCase() {
    // ToLower, ToUpper, ToTitle functions depend on equal size of both original and changed characters
    for (wchar32 i = 0; i != NUnicode::UnicodeInstancesLimit(); ++i) {
        UNIT_ASSERT(MustBeSurrogate(i) == MustBeSurrogate(ToLower(i)));
        UNIT_ASSERT(MustBeSurrogate(i) == MustBeSurrogate(ToUpper(i)));
        UNIT_ASSERT(MustBeSurrogate(i) == MustBeSurrogate(ToTitle(i)));
    }
}

void TConversionTest::TestUnicodeDetails() {
    TUtf16String temp;
    for (wchar32 i = 0; i != NUnicode::UnicodeInstancesLimit(); ++i) {
        temp.clear();
        WriteSymbol(i, temp);
        UNIT_ASSERT(temp.size() == W16SymbolSize(temp.c_str(), temp.c_str() + temp.size()));
    }
}

class TWideUtilTest: public TTestBase {
    UNIT_TEST_SUITE(TWideUtilTest);
    UNIT_TEST(TestCollapse);
    UNIT_TEST(TestCollapseBuffer);
    UNIT_TEST(TestStrip);
    UNIT_TEST(TestIsSpace);
    UNIT_TEST(TestEscapeHtmlChars);
    UNIT_TEST(TestToLower);
    UNIT_TEST(TestToUpper);
    UNIT_TEST(TestWideString);
    UNIT_TEST(TestCountWideChars);
    UNIT_TEST(TestIsValidUTF16);
    UNIT_TEST(TestIsStringASCII);
    UNIT_TEST(TestIsLowerWordStr);
    UNIT_TEST(TestIsUpperWordStr);
    UNIT_TEST(TestIsTitleStr);
    UNIT_TEST(TestIsLowerStr);
    UNIT_TEST(TestIsUpperStr);
    UNIT_TEST(TestToLowerStr);
    UNIT_TEST(TestToUpperStr);
    UNIT_TEST(TestToTitleStr);
    UNIT_TEST_SUITE_END();

public:
    void TestCollapse() {
        TUtf16String s;
        s.append(ws, Y_ARRAY_SIZE(ws)).append(3, 'a').append(ws, Y_ARRAY_SIZE(ws)).append(3, 'b').append(ws, Y_ARRAY_SIZE(ws));
        Collapse(s);
        UNIT_ASSERT(s == ASCIIToWide(" aaa bbb "));
        {
            const TUtf16String w(ASCIIToWide(" a b c "));
            s = w;
            Collapse(s);
            UNIT_ASSERT(s == w);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.c_str() == w.c_str()); // Collapse() does not change the string at all
#endif
        }
        s = ASCIIToWide("  123    456  ");
        Collapse(s);
        UNIT_ASSERT(s == ASCIIToWide(" 123 456 "));

        s = ASCIIToWide("  1\n\n\n23\t    4\f\f56  ");
        Collapse(s);
        UNIT_ASSERT(s == ASCIIToWide(" 1 23 4 56 "));

        s = ASCIIToWide(" 1\n\n\n\f\f56  ");
        Collapse(s);
        UNIT_ASSERT(s == ASCIIToWide(" 1 56 "));

        s = ASCIIToWide("  1\r\n,\n(\n23\t    4\f\f56  ");
        Collapse(s);
        UNIT_ASSERT(s == ASCIIToWide(" 1 , ( 23 4 56 "));

        s = ASCIIToWide("1 23  ");
        Collapse(s);
        UNIT_ASSERT(s == ASCIIToWide("1 23 "));
        {
            const TUtf16String w = ASCIIToWide(" ");
            s = w;
            Collapse(s);
            UNIT_ASSERT(s == w);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.c_str() == w.c_str()); // Collapse() does not change the string at all
#endif
        }
        s = ASCIIToWide("   ");
        Collapse(s);
        UNIT_ASSERT(s == ASCIIToWide(" "));

        s = ASCIIToWide(",\r\n\"");
        Collapse(s);
        UNIT_ASSERT(s == ASCIIToWide(", \""));

        s = ASCIIToWide("-");
        Collapse(s);
        UNIT_ASSERT(s == ASCIIToWide("-"));

        s.clear();
        Collapse(s);
        UNIT_ASSERT(s == TUtf16String());
    }

    void TestCollapseBuffer() {
        TUtf16String s;
        s.append(ws, Y_ARRAY_SIZE(ws)).append(3, 'a').append(ws, Y_ARRAY_SIZE(ws)).append(3, 'b').append(ws, Y_ARRAY_SIZE(ws));
        size_t n = Collapse(s.begin(), s.size());
        s.resize(n);
        UNIT_ASSERT(s == ASCIIToWide(" aaa bbb "));

        s = ASCIIToWide(" a b c ");
        n = Collapse(s.begin(), s.size());
        UNIT_ASSERT(n == s.size()); // length was not changed
        UNIT_ASSERT(s == ASCIIToWide(" a b c "));

        s = ASCIIToWide("  123    456  ");
        n = Collapse(s.begin(), s.size());
        s.resize(n);
        UNIT_ASSERT(s == ASCIIToWide(" 123 456 "));

        s = ASCIIToWide("  1\n\n\n23\t    4\f\f56  ");
        n = Collapse(s.begin(), s.size());
        s.resize(n);
        UNIT_ASSERT(s == ASCIIToWide(" 1 23 4 56 "));

        s = ASCIIToWide(" 1\n\n\n\f\f56  ");
        n = Collapse(s.begin(), s.size());
        s.resize(n);
        UNIT_ASSERT(s == ASCIIToWide(" 1 56 "));

        s = ASCIIToWide("  1\r\n,\n(\n23\t    4\f\f56  ");
        n = Collapse(s.begin(), s.size());
        s.resize(n);
        UNIT_ASSERT(s == ASCIIToWide(" 1 , ( 23 4 56 "));

        s = ASCIIToWide("1 23  ");
        n = Collapse(s.begin(), s.size());
        s.resize(n);
        UNIT_ASSERT(s == ASCIIToWide("1 23 "));

        s = ASCIIToWide(" ");
        n = Collapse(s.begin(), s.size());
        UNIT_ASSERT(n == 1);
        UNIT_ASSERT(s == ASCIIToWide(" "));

        s = ASCIIToWide("   ");
        n = Collapse(s.begin(), s.size());
        s.resize(n);
        UNIT_ASSERT(s == ASCIIToWide(" "));

        s = ASCIIToWide(",\r\n\"");
        n = Collapse(s.begin(), s.size());
        s.resize(n);
        UNIT_ASSERT(s == ASCIIToWide(", \""));

        s = ASCIIToWide("-");
        n = Collapse(s.begin(), s.size());
        UNIT_ASSERT(n == 1);
        UNIT_ASSERT(s == ASCIIToWide("-"));

        s = ASCIIToWide("\t");
        n = Collapse(s.begin(), s.size());
        UNIT_ASSERT(n == 1);
        UNIT_ASSERT(s == ASCIIToWide(" "));

        s.clear();
        n = Collapse(s.begin(), s.size());
        UNIT_ASSERT(n == 0);
        UNIT_ASSERT(s == TUtf16String());
    }

    void TestStrip() {
        TUtf16String s;

        Strip(s);
        UNIT_ASSERT(s == TUtf16String());
        StripLeft(s);
        UNIT_ASSERT(s == TUtf16String());
        StripRight(s);
        UNIT_ASSERT(s == TUtf16String());

        s = ASCIIToWide(" \t\r\n");
        Strip(s);
        UNIT_ASSERT(s == TUtf16String());
        s = ASCIIToWide(" \t\r\n");
        StripLeft(s);
        UNIT_ASSERT(s == TUtf16String());
        s = ASCIIToWide(" \t\r\n");
        StripRight(s);
        UNIT_ASSERT(s == TUtf16String());

        s = ASCIIToWide("\t\f\va \r\n");
        Strip(s);
        UNIT_ASSERT(s == ASCIIToWide("a"));
        s = ASCIIToWide("\t\f\va \r\n");
        StripLeft(s);
        UNIT_ASSERT(s == ASCIIToWide("a \r\n"));
        s = ASCIIToWide("\t\f\va \r\n");
        StripRight(s);
        UNIT_ASSERT(s == ASCIIToWide("\t\f\va"));

        s = ASCIIToWide("\r\na\r\nb\t\tc\r\n");
        Strip(s);
        UNIT_ASSERT(s == ASCIIToWide("a\r\nb\t\tc"));
        s = ASCIIToWide("\r\na\r\nb\t\tc\r\n");
        StripLeft(s);
        UNIT_ASSERT(s == ASCIIToWide("a\r\nb\t\tc\r\n"));
        s = ASCIIToWide("\r\na\r\nb\t\tc\r\n");
        StripRight(s);
        UNIT_ASSERT(s == ASCIIToWide("\r\na\r\nb\t\tc"));

        const TUtf16String w(ASCIIToWide("a  b"));
        s = w;
        Strip(s);
        UNIT_ASSERT(s == w);
#ifndef TSTRING_IS_STD_STRING
        UNIT_ASSERT(s.c_str() == w.c_str()); // Strip() does not change the string at all
#endif
        s = w;
        StripLeft(s);
        UNIT_ASSERT(s == w);
#ifndef TSTRING_IS_STD_STRING
        UNIT_ASSERT(s.c_str() == w.c_str()); // Strip() does not change the string at all
#endif
        s = w;
        StripRight(s);
        UNIT_ASSERT(s == w);
#ifndef TSTRING_IS_STD_STRING
        UNIT_ASSERT(s.c_str() == w.c_str()); // Strip() does not change the string at all
#endif
    }

    void TestIsSpace() {
        UNIT_ASSERT(!IsSpace(TUtf16String()));

        UNIT_ASSERT(IsSpace(ws, Y_ARRAY_SIZE(ws)));

        TUtf16String w;
        w.assign(ws, Y_ARRAY_SIZE(ws)).append(TUtf16String(1, '!'));
        UNIT_ASSERT(!IsSpace(w.c_str(), w.size()));

        w.assign(TUtf16String(1, '_')).append(ws, Y_ARRAY_SIZE(ws));
        UNIT_ASSERT(!IsSpace(w.c_str(), w.size()));

        w.assign(ws, Y_ARRAY_SIZE(ws)).append(TUtf16String(1, '$')).append(ws, Y_ARRAY_SIZE(ws));
        UNIT_ASSERT(!IsSpace(w.c_str(), w.size()));
    }

    void TestEscapeHtmlChars() {
        // characters from the first half of the ASCII table
        for (wchar16 c = 1; c < 0x7F; ++c) {
            TUtf16String w(1, c);
            EscapeHtmlChars<false>(w);

            switch (c) {
                case '<':
                    UNIT_ASSERT(w == ASCIIToWide("&lt;"));
                    break;
                case '>':
                    UNIT_ASSERT(w == ASCIIToWide("&gt;"));
                    break;
                case '&':
                    UNIT_ASSERT(w == ASCIIToWide("&amp;"));
                    break;
                case '"':
                    UNIT_ASSERT(w == ASCIIToWide("&quot;"));
                    break;
                default:
                    UNIT_ASSERT(w == TUtf16String(1, c));
                    break;
            }
        }

        for (wchar16 c = 1; c < 0x7F; ++c) {
            TUtf16String w(1, c);
            EscapeHtmlChars<true>(w);

            switch (c) {
                case '<':
                    UNIT_ASSERT(w == ASCIIToWide("&lt;"));
                    break;
                case '>':
                    UNIT_ASSERT(w == ASCIIToWide("&gt;"));
                    break;
                case '&':
                    UNIT_ASSERT(w == ASCIIToWide("&amp;"));
                    break;
                case '"':
                    UNIT_ASSERT(w == ASCIIToWide("&quot;"));
                    break;
                case '\r':
                case '\n':
                    UNIT_ASSERT(w == ASCIIToWide("<BR>"));
                    break;
                default:
                    UNIT_ASSERT(w == TUtf16String(1, c));
                    break;
            }
        }
    }

    void TestToLower() {
        const size_t n = 32;
        wchar16 upperCase[n];
        std::copy(wideCyrillicAlphabet, wideCyrillicAlphabet + n, upperCase);
        ToLower(upperCase, n);
        UNIT_ASSERT(TWtringBuf(upperCase, n) == TWtringBuf(wideCyrillicAlphabet + n, n));
    }

    void TestToUpper() {
        const size_t n = 32;
        wchar16 lowerCase[n];
        std::copy(wideCyrillicAlphabet + n, wideCyrillicAlphabet + n * 2, lowerCase);
        ToUpper(lowerCase, n);
        UNIT_ASSERT(TWtringBuf(lowerCase, n) == TWtringBuf(wideCyrillicAlphabet, n));
    }

    void TestWideString() {
        const TUtf16String original = UTF32ToWide(WideStringTestData[0], CaseTestDataSize);
        const TUtf16String lower = UTF32ToWide(WideStringTestData[1], CaseTestDataSize);
        const TUtf16String upper = UTF32ToWide(WideStringTestData[2], CaseTestDataSize);
        const TUtf16String title = UTF32ToWide(WideStringTestData[3], CaseTestDataSize);
        TUtf16String temp;

        temp = original;
        temp.to_lower();
        UNIT_ASSERT(temp == lower);

        temp = original;
        ToLower(temp.begin(), temp.size());
        UNIT_ASSERT(temp == lower);

        temp = original;
        temp.to_upper();
        UNIT_ASSERT(temp == upper);

        temp = original;
        ToUpper(temp.begin(), temp.size());
        UNIT_ASSERT(temp == upper);

        temp = original;
        temp.to_title();
        UNIT_ASSERT(temp == title);

        temp = original;
        ToTitle(temp.begin(), temp.size());
        UNIT_ASSERT(temp == title);

        TVector<wchar32> buffer(WideStringTestData[0], WideStringTestData[0] + CaseTestDataSize);
        std::reverse(buffer.begin(), buffer.end());
        const TUtf16String reversed = UTF32ToWide(buffer.data(), buffer.size());

        temp = original;
        ReverseInPlace(temp);
        UNIT_ASSERT(temp == reversed);
    }

    void TestCountWideChars() {
        UNIT_ASSERT_EQUAL(CountWideChars(UTF8ToWide("привет!")), 7);
        TUtf16String wideStr = UTF8ToWide("\xf0\x9f\x92\xb8привет!");
        UNIT_ASSERT_EQUAL(wideStr.size(), 9);
        UNIT_ASSERT_EQUAL(CountWideChars(wideStr), 8);
    }

    void TestIsValidUTF16() {
        static wchar16 str1[] = {'h', 'e', 'l', 'l', 'o', '!', 0};
        static wchar16 str2[] = {'h', 'e', 'l', 'l', 'o', 0xD842, 0xDEAD, '!', 0};
        static wchar16 str3[] = {'h', 'e', 'l', 'l', 'o', 0xD842, '!', 0};
        static wchar16 str4[] = {'h', 'e', 'l', 'l', 'o', 0xDEAD, 0xD842, '!', 0};
        static wchar16 str5[] = {'h', 'e', 'l', 'l', 'o', 0xD842, 0xDEAD, 0xDEAD, '!', 0};
        UNIT_ASSERT(IsValidUTF16(TWtringBuf(str1)));
        UNIT_ASSERT(IsValidUTF16(TWtringBuf(str2)));
        UNIT_ASSERT(!IsValidUTF16(TWtringBuf(str3)));
        UNIT_ASSERT(!IsValidUTF16(TWtringBuf(str4)));
        UNIT_ASSERT(!IsValidUTF16(TWtringBuf(str5)));
    }

    void TestIsStringASCII() {
        static char charAscii[] = "0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF";
        static wchar16 char16Ascii[] = {
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A',
            'B', 'C', 'D', 'E', 'F', '0', '1', '2', '3', '4', '5', '6',
            '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 0};

        // Test a variety of the fragment start positions and lengths in order to make
        // sure that bit masking in IsStringASCII works correctly.
        // Also, test that a non-ASCII character will be detected regardless of its
        // position inside the string.
        {
            const size_t stringLength = Y_ARRAY_SIZE(charAscii) - 1;
            for (size_t offset = 0; offset < 8; ++offset) {
                for (size_t len = 0, maxLen = stringLength - offset; len < maxLen; ++len) {
                    UNIT_ASSERT(IsStringASCII(charAscii + offset, charAscii + offset + len));
                    for (size_t charPos = offset; charPos < len; ++charPos) {
                        charAscii[charPos] |= '\x80';
                        UNIT_ASSERT(!IsStringASCII(charAscii + offset, charAscii + offset + len));
                        charAscii[charPos] &= ~'\x80';
                    }
                }
            }
        }

        {
            const size_t stringLength = Y_ARRAY_SIZE(char16Ascii) - 1;
            for (size_t offset = 0; offset < 4; ++offset) {
                for (size_t len = 0, maxLen = stringLength - offset; len < maxLen; ++len) {
                    UNIT_ASSERT(IsStringASCII(char16Ascii + offset, char16Ascii + offset + len));

                    for (size_t charPos = offset; charPos < len; ++charPos) {
                        char16Ascii[charPos] |= 0x80;
                        UNIT_ASSERT(
                            !IsStringASCII(char16Ascii + offset, char16Ascii + offset + len));

                        char16Ascii[charPos] &= ~0x80;
                        // Also test when the upper half is non-zero.
                        char16Ascii[charPos] |= 0x100;
                        UNIT_ASSERT(
                            !IsStringASCII(char16Ascii + offset, char16Ascii + offset + len));
                        char16Ascii[charPos] &= ~0x100;
                    }
                }
            }
        }
    }

    void TestIsLowerWordStr() {
        UNIT_ASSERT(IsLowerWord(TWtringBuf()));
        UNIT_ASSERT(IsLowerWord(UTF8ToWide("")));
        UNIT_ASSERT(IsLowerWord(UTF8ToWide("test")));
        UNIT_ASSERT(IsLowerWord(UTF8ToWide("тест"))); // "тест" is "test" in russian (cyrrilic)
        UNIT_ASSERT(!IsLowerWord(UTF8ToWide("тест тест")));
        UNIT_ASSERT(!IsLowerWord(UTF8ToWide("тест100500")));

        UNIT_ASSERT(!IsLowerWord(UTF8ToWide("Test")));
        UNIT_ASSERT(!IsLowerWord(UTF8ToWide("tesT")));
        UNIT_ASSERT(!IsLowerWord(UTF8ToWide("tEst")));

        UNIT_ASSERT(!IsLowerWord(UTF8ToWide("Тест")));
        UNIT_ASSERT(!IsLowerWord(UTF8ToWide("теСт")));
        UNIT_ASSERT(!IsLowerWord(UTF8ToWide("тесТ")));
    }

    void TestIsUpperWordStr() {
        UNIT_ASSERT(IsUpperWord(TWtringBuf()));
        UNIT_ASSERT(IsUpperWord(UTF8ToWide("")));
        UNIT_ASSERT(IsUpperWord(UTF8ToWide("TEST")));
        UNIT_ASSERT(IsUpperWord(UTF8ToWide("ТЕСТ")));
        UNIT_ASSERT(!IsUpperWord(UTF8ToWide("тест тест")));
        UNIT_ASSERT(!IsUpperWord(UTF8ToWide("тест100500")));

        UNIT_ASSERT(!IsUpperWord(UTF8ToWide("Test")));
        UNIT_ASSERT(!IsUpperWord(UTF8ToWide("tesT")));
        UNIT_ASSERT(!IsUpperWord(UTF8ToWide("tEst")));

        UNIT_ASSERT(!IsUpperWord(UTF8ToWide("Тест")));
        UNIT_ASSERT(!IsUpperWord(UTF8ToWide("теСт")));
        UNIT_ASSERT(!IsUpperWord(UTF8ToWide("тесТ")));
    }

    void TestIsTitleStr() {
        UNIT_ASSERT(!IsTitleWord(TWtringBuf()));
        UNIT_ASSERT(!IsTitleWord(UTF8ToWide("")));
        UNIT_ASSERT(!IsTitleWord(UTF8ToWide("t")));
        UNIT_ASSERT(!IsTitleWord(UTF8ToWide("й")));
        UNIT_ASSERT(IsTitleWord(UTF8ToWide("T")));
        UNIT_ASSERT(IsTitleWord(UTF8ToWide("Й")));
        UNIT_ASSERT(IsTitleWord(UTF8ToWide("Test")));
        UNIT_ASSERT(IsTitleWord(UTF8ToWide("Тест")));
        UNIT_ASSERT(!IsTitleWord(UTF8ToWide("тест тест")));
        UNIT_ASSERT(!IsTitleWord(UTF8ToWide("тест100500")));
        UNIT_ASSERT(!IsTitleWord(UTF8ToWide("Тест тест")));
        UNIT_ASSERT(!IsTitleWord(UTF8ToWide("Тест100500")));

        UNIT_ASSERT(!IsTitleWord(UTF8ToWide("tesT")));
        UNIT_ASSERT(!IsTitleWord(UTF8ToWide("tEst")));

        UNIT_ASSERT(!IsTitleWord(UTF8ToWide("теСт")));
        UNIT_ASSERT(!IsTitleWord(UTF8ToWide("тесТ")));
    }

    void TestIsLowerStr() {
        UNIT_ASSERT(IsLower(TWtringBuf()));
        UNIT_ASSERT(IsLower(UTF8ToWide("")));
        UNIT_ASSERT(IsLower(UTF8ToWide("test")));
        UNIT_ASSERT(IsLower(UTF8ToWide("тест"))); // "тест" is "test" in russian (cyrrilic)
        UNIT_ASSERT(IsLower(UTF8ToWide("тест тест")));
        UNIT_ASSERT(IsLower(UTF8ToWide("тест100500")));

        UNIT_ASSERT(!IsLower(UTF8ToWide("Test")));
        UNIT_ASSERT(!IsLower(UTF8ToWide("tesT")));
        UNIT_ASSERT(!IsLower(UTF8ToWide("tEst")));

        UNIT_ASSERT(!IsLower(UTF8ToWide("Тест")));
        UNIT_ASSERT(!IsLower(UTF8ToWide("теСт")));
        UNIT_ASSERT(!IsLower(UTF8ToWide("тесТ")));
    }

    void TestIsUpperStr() {
        UNIT_ASSERT(IsUpper(TWtringBuf()));
        UNIT_ASSERT(IsUpper(UTF8ToWide("")));
        UNIT_ASSERT(IsUpper(UTF8ToWide("TEST")));
        UNIT_ASSERT(IsUpper(UTF8ToWide("ТЕСТ")));
        UNIT_ASSERT(IsUpper(UTF8ToWide("ТЕСТ ТЕСТ")));
        UNIT_ASSERT(IsUpper(UTF8ToWide("ТЕСТ100500")));

        UNIT_ASSERT(!IsUpper(UTF8ToWide("Test")));
        UNIT_ASSERT(!IsUpper(UTF8ToWide("tesT")));
        UNIT_ASSERT(!IsUpper(UTF8ToWide("tEst")));

        UNIT_ASSERT(!IsUpper(UTF8ToWide("Тест")));
        UNIT_ASSERT(!IsUpper(UTF8ToWide("теСт")));
        UNIT_ASSERT(!IsUpper(UTF8ToWide("тесТ")));
    }

    void TestToLowerStr() {
        // In these test and test for `ToUpper` and `ToTitle` we are checking that string keep
        // pointing to the same piece of memory we are doing it the following way:
        //
        // TUtf16String s = ...
        // const auto copy = s;
        // ...
        // UNIT_ASSERT(s.data() == copy.data())
        //
        // It saves us a couple lines (we are reusing `copy` later) and if one day `TString` will
        // become non-refcounted we'll need to rewrite it to something like:
        //
        // TUtf16String s = ...
        // const auto* const data = s.data();
        // const auto length = s.length();
        // ...
        // UNIT_ASSERT(s.data() == data);
        // UNIT_ASSERT(s.length() == length);
        {
            TUtf16String s;
            auto writableCopy = s;
            const auto copy = s;
            const TUtf16String lower;

            UNIT_ASSERT(!ToLower(s));
            UNIT_ASSERT(s == lower);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(!ToLower(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == lower);

            UNIT_ASSERT(!ToLower(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == lower);

            UNIT_ASSERT(ToLowerRet(copy) == lower);
            UNIT_ASSERT(ToLowerRet(TWtringBuf(copy)) == lower);
        }
        {
            TUtf16String s = UTF8ToWide("");
            auto writableCopy = s;
            const auto copy = s;
            const TUtf16String lower;

            UNIT_ASSERT(!ToLower(s));
            UNIT_ASSERT(s == lower);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(!ToLower(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == lower);

            UNIT_ASSERT(!ToLower(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == lower);

            UNIT_ASSERT(ToLowerRet(copy) == lower);
            UNIT_ASSERT(ToLowerRet(TWtringBuf(copy)) == lower);
        }
        {
            TUtf16String s;
            const auto copy = s;
            const TUtf16String lower;

            UNIT_ASSERT(!ToLower(s, 100500));
            UNIT_ASSERT(s == lower);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(ToLowerRet(copy, 100500) == lower);
            UNIT_ASSERT(ToLowerRet(TWtringBuf(copy), 100500) == lower);
        }
        {
            TUtf16String s;
            const auto copy = s;
            const TUtf16String lower;

            UNIT_ASSERT(!ToLower(s, 100500, 1111));
            UNIT_ASSERT(s == lower);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(ToLowerRet(copy, 100500, 1111) == lower);
            UNIT_ASSERT(ToLowerRet(TWtringBuf(copy), 100500, 1111) == lower);
        }
        {
            auto s = UTF8ToWide("Й");
            auto writableCopy = s;
            const auto copy = s;
            const auto lower = UTF8ToWide("й");

            UNIT_ASSERT(ToLower(s));
            UNIT_ASSERT(s == lower);

            UNIT_ASSERT(ToLower(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == lower);

            UNIT_ASSERT(ToLower(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == lower);

            UNIT_ASSERT(ToLowerRet(copy) == lower);
            UNIT_ASSERT(ToLowerRet(TWtringBuf(copy)) == lower);
        }
        {
            auto s = UTF8ToWide("й");
            auto writableCopy = s;
            const auto copy = s;
            const auto lower = UTF8ToWide("й");

            UNIT_ASSERT(!ToLower(s));
            UNIT_ASSERT(s == lower);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(!ToLower(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == lower);

            UNIT_ASSERT(!ToLower(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == lower);

            UNIT_ASSERT(ToLowerRet(copy) == lower);
            UNIT_ASSERT(ToLowerRet(TWtringBuf(copy)) == lower);
        }
        {
            auto s = UTF8ToWide("тест");
            auto writableCopy = s;
            const auto copy = s;
            const auto lower = UTF8ToWide("тест");

            UNIT_ASSERT(!ToLower(s));
            UNIT_ASSERT(s == lower);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(!ToLower(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == lower);

            UNIT_ASSERT(!ToLower(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == lower);

            UNIT_ASSERT(ToLowerRet(copy) == lower);
            UNIT_ASSERT(ToLowerRet(TWtringBuf(copy)) == lower);
        }
        {
            auto s = UTF8ToWide("Тест");
            auto writableCopy = s;
            const auto copy = s;
            const auto lower = UTF8ToWide("тест");

            UNIT_ASSERT(ToLower(s));
            UNIT_ASSERT(s == lower);

            UNIT_ASSERT(ToLower(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == lower);

            UNIT_ASSERT(ToLower(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == lower);

            UNIT_ASSERT(ToLowerRet(copy) == lower);
            UNIT_ASSERT(ToLowerRet(TWtringBuf(copy)) == lower);
        }
        {
            TUtf16String s = UTF8ToWide("тЕст");
            const auto copy = s;
            const auto lower = UTF8ToWide("тест");

            UNIT_ASSERT(ToLower(s));
            UNIT_ASSERT(s == UTF8ToWide("тест"));

            UNIT_ASSERT(ToLowerRet(copy) == lower);
            UNIT_ASSERT(ToLowerRet(TWtringBuf(copy)) == lower);
        }
        {
            auto s = UTF8ToWide("тЕст");
            const auto copy = s;
            const auto lower = UTF8ToWide("тЕст");

            UNIT_ASSERT(!ToLower(s, 2));
            UNIT_ASSERT(s == lower);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(ToLowerRet(copy, 2) == lower);
            UNIT_ASSERT(ToLowerRet(TWtringBuf(copy), 2) == lower);
        }
        {
            auto s = UTF8ToWide("теСт");
            const auto copy = s;
            const auto lower = UTF8ToWide("тест");

            UNIT_ASSERT(ToLower(s, 2));
            UNIT_ASSERT(s == lower);

            UNIT_ASSERT(ToLowerRet(copy, 2) == lower);
            UNIT_ASSERT(ToLowerRet(TWtringBuf(copy), 2) == lower);
        }
        {
            auto s = UTF8ToWide("теСт");
            const auto copy = s;
            const auto lower = UTF8ToWide("теСт");

            UNIT_ASSERT(!ToLower(s, 3, 1));
            UNIT_ASSERT(s == copy);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(ToLowerRet(copy, 3, 1) == lower);
            UNIT_ASSERT(ToLowerRet(TWtringBuf(copy), 3, 1) == lower);
        }
        {
            auto s = UTF8ToWide("теСт");
            const auto copy = s;
            const auto lower = UTF8ToWide("теСт");

            UNIT_ASSERT(!ToLower(s, 3, 100500));
            UNIT_ASSERT(s == copy);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(ToLowerRet(copy, 3, 100500) == lower);
            UNIT_ASSERT(ToLowerRet(TWtringBuf(copy), 3, 100500) == lower);
        }
    }

    void TestToUpperStr() {
        {
            TUtf16String s;
            auto writableCopy = s;
            const auto copy = s;
            const TUtf16String upper;

            UNIT_ASSERT(!ToUpper(s));
            UNIT_ASSERT(s == upper);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(!ToUpper(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(!ToUpper(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(ToUpperRet(copy) == upper);
            UNIT_ASSERT(ToUpperRet(TWtringBuf(copy)) == upper);
        }
        {
            auto s = UTF8ToWide("");
            auto writableCopy = s;
            const auto copy = s;
            const TUtf16String upper;

            UNIT_ASSERT(!ToUpper(s));
            UNIT_ASSERT(s == upper);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(!ToUpper(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(!ToUpper(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(ToUpperRet(copy) == upper);
            UNIT_ASSERT(ToUpperRet(TWtringBuf(copy)) == upper);
        }
        {
            TUtf16String s;
            auto writableCopy = s;
            const auto copy = s;
            const TUtf16String upper;

            UNIT_ASSERT(!ToUpper(s, 100500));
            UNIT_ASSERT(s == upper);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(!ToUpper(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(!ToUpper(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(ToUpperRet(copy, 100500) == upper);
            UNIT_ASSERT(ToUpperRet(TWtringBuf(copy), 100500) == upper);
        }
        {
            TUtf16String s;
            const auto copy = s;
            const TUtf16String upper;

            UNIT_ASSERT(!ToUpper(s, 100500, 1111));
            UNIT_ASSERT(s == upper);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(ToUpperRet(copy, 100500, 1111) == upper);
            UNIT_ASSERT(ToUpperRet(TWtringBuf(copy), 100500, 1111) == upper);
        }
        {
            auto s = UTF8ToWide("й");
            auto writableCopy = s;
            const auto copy = s;
            const auto upper = UTF8ToWide("Й");

            UNIT_ASSERT(ToUpper(s));
            UNIT_ASSERT(s == upper);

            UNIT_ASSERT(ToUpper(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(ToUpper(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(ToUpperRet(copy) == upper);
            UNIT_ASSERT(ToUpperRet(TWtringBuf(copy)) == upper);
        }
        {
            auto s = UTF8ToWide("Й");
            auto writableCopy = s;
            const auto copy = s;
            const auto upper = UTF8ToWide("Й");

            UNIT_ASSERT(!ToUpper(s));
            UNIT_ASSERT(s == copy);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(!ToUpper(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(!ToUpper(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(ToUpperRet(copy) == upper);
            UNIT_ASSERT(ToUpperRet(TWtringBuf(copy)) == upper);
        }
        {
            auto s = UTF8ToWide("тест");
            auto writableCopy = s;
            const auto copy = s;
            const auto upper = UTF8ToWide("ТЕСТ");

            UNIT_ASSERT(ToUpper(s));
            UNIT_ASSERT(s == upper);

            UNIT_ASSERT(ToUpper(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(ToUpper(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(ToUpperRet(copy) == upper);
            UNIT_ASSERT(ToUpperRet(TWtringBuf(copy)) == upper);
        }
        {
            auto s = UTF8ToWide("Тест");
            auto writableCopy = s;
            const auto copy = s;
            const auto upper = UTF8ToWide("ТЕСТ");

            UNIT_ASSERT(ToUpper(s));
            UNIT_ASSERT(s == upper);

            UNIT_ASSERT(ToUpper(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(ToUpper(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(ToUpperRet(copy) == upper);
            UNIT_ASSERT(ToUpperRet(TWtringBuf(copy)) == upper);
        }
        {
            auto s = UTF8ToWide("тЕст");
            auto writableCopy = s;
            const auto copy = s;
            const auto upper = UTF8ToWide("ТЕСТ");

            UNIT_ASSERT(ToUpper(s));
            UNIT_ASSERT(s == upper);

            UNIT_ASSERT(ToUpper(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(ToUpper(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == upper);

            UNIT_ASSERT(ToUpperRet(copy) == upper);
            UNIT_ASSERT(ToUpperRet(TWtringBuf(copy)) == upper);
        }
        {
            auto s = UTF8ToWide("тЕст");
            const auto copy = s;
            const auto upper = UTF8ToWide("тЕСТ");

            UNIT_ASSERT(ToUpper(s, 2));
            UNIT_ASSERT(s == upper);

            UNIT_ASSERT(ToUpperRet(copy, 2) == upper);
            UNIT_ASSERT(ToUpperRet(TWtringBuf(copy), 2) == upper);
        }
        {
            auto s = UTF8ToWide("теСт");
            const auto copy = s;
            const auto upper = UTF8ToWide("теСТ");

            UNIT_ASSERT(ToUpper(s, 2));
            UNIT_ASSERT(s == upper);

            UNIT_ASSERT(ToUpperRet(copy, 2) == upper);
            UNIT_ASSERT(ToUpperRet(TWtringBuf(copy), 2) == upper);
        }
        {
            auto s = UTF8ToWide("теСт");
            const auto copy = s;
            const auto upper = UTF8ToWide("теСТ");

            UNIT_ASSERT(ToUpper(s, 3, 1));
            UNIT_ASSERT(s == upper);

            UNIT_ASSERT(ToUpperRet(copy, 3, 1) == upper);
            UNIT_ASSERT(ToUpperRet(TWtringBuf(copy), 3, 1) == upper);
        }
        {
            auto s = UTF8ToWide("теСт");
            const auto copy = s;
            const auto upper = UTF8ToWide("теСТ");

            UNIT_ASSERT(ToUpper(s, 3, 100500));
            UNIT_ASSERT(s == upper);

            UNIT_ASSERT(ToUpperRet(copy, 3, 100500) == upper);
            UNIT_ASSERT(ToUpperRet(TWtringBuf(copy), 3, 100500) == upper);
        }
    }

    void TestToTitleStr() {
        {
            TUtf16String s;
            auto writableCopy = s;
            const auto copy = s;
            const TUtf16String title;

            UNIT_ASSERT(!ToTitle(s));
            UNIT_ASSERT(s == title);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(!ToTitle(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(!ToTitle(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(ToTitleRet(copy) == title);
            UNIT_ASSERT(ToTitleRet(TWtringBuf(copy)) == title);
        }
        {
            auto s = UTF8ToWide("");
            auto writableCopy = s;
            const auto copy = s;
            const TUtf16String title;

            UNIT_ASSERT(!ToTitle(s));
            UNIT_ASSERT(s == title);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(!ToTitle(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(!ToTitle(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(ToTitleRet(copy) == title);
            UNIT_ASSERT(ToTitleRet(TWtringBuf(copy)) == title);
        }
        {
            TUtf16String s;
            const auto copy = s;
            const TUtf16String title;

            UNIT_ASSERT(!ToTitle(s, 100500));
            UNIT_ASSERT(s == title);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(ToTitleRet(copy) == title);
            UNIT_ASSERT(ToTitleRet(TWtringBuf(copy)) == title);
        }
        {
            TUtf16String s;
            const auto copy = s;
            const TUtf16String title;

            UNIT_ASSERT(!ToTitle(s, 100500, 1111));
            UNIT_ASSERT(s == title);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(ToTitleRet(copy) == title);
            UNIT_ASSERT(ToTitleRet(TWtringBuf(copy)) == title);
        }
        {
            auto s = UTF8ToWide("й");
            auto writableCopy = s;
            const auto copy = s;
            const auto title = UTF8ToWide("Й");

            UNIT_ASSERT(ToTitle(s));
            UNIT_ASSERT(s == title);

            UNIT_ASSERT(ToTitle(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(ToTitle(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(ToTitleRet(copy) == title);
            UNIT_ASSERT(ToTitleRet(TWtringBuf(copy)) == title);
        }
        {
            auto s = UTF8ToWide("Й");
            auto writableCopy = s;
            const auto copy = s;
            const auto title = UTF8ToWide("Й");

            UNIT_ASSERT(!ToTitle(s));
            UNIT_ASSERT(s == title);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(!ToTitle(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(!ToTitle(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(ToTitleRet(copy) == title);
            UNIT_ASSERT(ToTitleRet(TWtringBuf(copy)) == title);
        }
        {
            auto s = UTF8ToWide("тест");
            auto writableCopy = s;
            const auto copy = s;
            const auto title = UTF8ToWide("Тест");

            UNIT_ASSERT(ToTitle(s));
            UNIT_ASSERT(s == title);

            UNIT_ASSERT(ToTitle(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(ToTitle(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(ToTitleRet(copy) == title);
            UNIT_ASSERT(ToTitleRet(TWtringBuf(copy)) == title);
        }
        {
            auto s = UTF8ToWide("Тест");
            auto writableCopy = s;
            const auto copy = s;
            const auto title = UTF8ToWide("Тест");

            UNIT_ASSERT(!ToTitle(s));
            UNIT_ASSERT(s == title);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(!ToTitle(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(!ToTitle(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(ToTitleRet(copy) == title);
            UNIT_ASSERT(ToTitleRet(TWtringBuf(copy)) == title);
        }
        {
            auto s = UTF8ToWide("тЕст");
            auto writableCopy = s;
            const auto copy = s;
            const auto title = UTF8ToWide("Тест");

            UNIT_ASSERT(ToTitle(s));
            UNIT_ASSERT(s == title);

            UNIT_ASSERT(ToTitle(writableCopy.Detach(), writableCopy.size()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(ToTitle(copy.data(), copy.size(), writableCopy.Detach()));
            UNIT_ASSERT(writableCopy == title);

            UNIT_ASSERT(ToTitleRet(copy) == title);
            UNIT_ASSERT(ToTitleRet(TWtringBuf(copy)) == title);
        }
        {
            auto s = UTF8ToWide("тЕст");
            const auto copy = s;
            const auto title = UTF8ToWide("тЕСт");

            UNIT_ASSERT(ToTitle(s, 2));
            UNIT_ASSERT(s == title);

            UNIT_ASSERT(ToTitleRet(copy, 2) == title);
            UNIT_ASSERT(ToTitleRet(TWtringBuf(copy), 2) == title);
        }
        {
            auto s = UTF8ToWide("теСт");
            const auto copy = s;
            const auto title = UTF8ToWide("теСт");

            UNIT_ASSERT(!ToTitle(s, 2));
            UNIT_ASSERT(s == title);
#ifndef TSTRING_IS_STD_STRING
            UNIT_ASSERT(s.data() == copy.data());
#endif

            UNIT_ASSERT(ToTitleRet(copy, 2) == title);
            UNIT_ASSERT(ToTitleRet(TWtringBuf(copy), 2) == title);
        }
        {
            auto s = UTF8ToWide("теСт");
            const auto copy = s;
            const auto title = UTF8ToWide("теСТ");

            UNIT_ASSERT(ToTitle(s, 3, 1));
            UNIT_ASSERT(s == title);

            UNIT_ASSERT(ToTitleRet(copy, 3, 1) == title);
            UNIT_ASSERT(ToTitleRet(TWtringBuf(copy), 3, 1) == title);
        }
        {
            auto s = UTF8ToWide("теСт");
            const auto copy = s;
            const auto title = UTF8ToWide("теСТ");

            UNIT_ASSERT(ToTitle(s, 3, 100500));
            UNIT_ASSERT(s == title);

            UNIT_ASSERT(ToTitleRet(copy, 3, 100500) == title);
            UNIT_ASSERT(ToTitleRet(TWtringBuf(copy), 3, 100500) == title);
        }
    }
};

UNIT_TEST_SUITE_REGISTRATION(TWideUtilTest);
