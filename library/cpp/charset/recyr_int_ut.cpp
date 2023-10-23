#include "codepage.h"
#include "recyr.hh"
#include "wide.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/charset/utf8.h>
#include <util/system/yassert.h>

class TRecyr_intTest: public TTestBase {
private:
    UNIT_TEST_SUITE(TRecyr_intTest);
    UNIT_TEST(TestUTFFromUnknownPlane);
    UNIT_TEST(TestBrokenMultibyte);
    UNIT_TEST(TestSurrogatePairs);
    UNIT_TEST_SUITE_END();

public:
    void TestUTFFromUnknownPlane();
    void TestBrokenMultibyte();
    void TestSurrogatePairs();
};

void TRecyr_intTest::TestBrokenMultibyte() {
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

void TRecyr_intTest::TestUTFFromUnknownPlane() {
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

void TRecyr_intTest::TestSurrogatePairs() {
    const char* utf8NonBMP = "\xf4\x80\x89\x84\xf4\x80\x89\x87\xf4\x80\x88\xba";
    wchar16 wNonBMPDummy[] = {0xDBC0, 0xDE44, 0xDBC0, 0xDE47, 0xDBC0, 0xDE3A};
    TestSurrogates(utf8NonBMP, wNonBMPDummy, Y_ARRAY_SIZE(wNonBMPDummy));

    const char* utf8NonBMP2 = "ab\xf4\x80\x89\x87n";
    wchar16 wNonBMPDummy2[] = {'a', 'b', 0xDBC0, 0xDE47, 'n'};
    TestSurrogates(utf8NonBMP2, wNonBMPDummy2, Y_ARRAY_SIZE(wNonBMPDummy2));
}
