#include "wide.h"
#include "recyr.hh"
#include "codepage.h"

#include <library/cpp/testing/unittest/registar.h>

static void TestIconv(const TString& utf8, const TString& other, ECharset enc) {
    TUtf16String wide0 = CharToWide(utf8, CODES_UTF8);
    TUtf16String wide1 = CharToWide(other, enc);

    UNIT_ASSERT(wide0 == wide1);

    TString temp = WideToUTF8(wide0);
    UNIT_ASSERT(temp == utf8);

    temp = WideToChar(wide0, enc);
    UNIT_ASSERT(temp == other);

    temp = Recode(enc, CODES_UTF8, other);
    UNIT_ASSERT(temp == utf8);

    temp = Recode(CODES_UTF8, enc, utf8);
    UNIT_ASSERT(temp == other);

    size_t read = 0;
    size_t written = 0;

    RECODE_RESULT res = RecodeToUnicode(enc, other.c_str(), wide1.begin(), other.size(), wide1.size(), read, written);
    UNIT_ASSERT(res == RECODE_OK);
    UNIT_ASSERT(read == other.size());
    UNIT_ASSERT(written == wide1.size());
    UNIT_ASSERT(wide0 == wide1);

    res = RecodeFromUnicode(enc, wide0.c_str(), temp.begin(), wide0.size(), temp.size(), read, written);
    UNIT_ASSERT(res == RECODE_OK);
    UNIT_ASSERT(read == wide0.size());
    UNIT_ASSERT(written == other.size());
    UNIT_ASSERT(temp == other);
}

class TIconvTest: public TTestBase {
    static void TestSurrogates(const char* str, const wchar16* wide, size_t wideSize) {
        size_t sSize = strlen(str);
        size_t wSize = sSize * 2;
        TArrayHolder<wchar16> w(new wchar16[wSize]);

        size_t read = 0;
        size_t written = 0;
        NICONVPrivate::RecodeToUnicode(CODES_UTF8, str, w.Get(), sSize, wSize, read, written);
        UNIT_ASSERT(read == sSize);
        UNIT_ASSERT(written == wideSize);
        UNIT_ASSERT(!memcmp(w.Get(), wide, wideSize));

        TArrayHolder<char> s(new char[sSize]);
        NICONVPrivate::RecodeFromUnicode(CODES_UTF8, w.Get(), s.Get(), wideSize, sSize, read, written);
        UNIT_ASSERT(read == wideSize);
        UNIT_ASSERT(written == sSize);
        UNIT_ASSERT(!memcmp(s.Get(), str, sSize));
    }

private:
    UNIT_TEST_SUITE(TIconvTest);
    UNIT_TEST(TestBig5);
    UNIT_TEST(TestSurrogatePairs);
    UNIT_TEST_SUITE_END();

public:
    void TestBig5() {
        UNIT_ASSERT(!NCodepagePrivate::NativeCodepage(CODES_BIG5));
        const char* UTF8 = "\xe5\xad\xb8\xe7\x94\x9f\xe7\xb8\xbd\xe4\xba\xba\xe6\x95\xb8\xe6\x99\xae\xe9\x80\x9a\xe7\x8f\xad";
        const char* BIG5 = "\xbe\xc7\xa5\xcd\xc1\x60\xa4\x48\xbc\xc6\xb4\xb6\xb3\x71\xaf\x5a";

        TestIconv(UTF8, BIG5, CODES_BIG5);
    }

    void TestSurrogatePairs() {
        const char* utf8NonBMP = "\xf4\x80\x89\x84\xf4\x80\x89\x87\xf4\x80\x88\xba";
        wchar16 wNonBMPDummy[] = {0xDBC0, 0xDE44, 0xDBC0, 0xDE47, 0xDBC0, 0xDE3A};
        TestSurrogates(utf8NonBMP, wNonBMPDummy, Y_ARRAY_SIZE(wNonBMPDummy));

        const char* utf8NonBMP2 = "ab\xf4\x80\x89\x87n";
        wchar16 wNonBMPDummy2[] = {'a', 'b', 0xDBC0, 0xDE47, 'n'};
        TestSurrogates(utf8NonBMP2, wNonBMPDummy2, Y_ARRAY_SIZE(wNonBMPDummy2));
    }
};

UNIT_TEST_SUITE_REGISTRATION(TIconvTest);
