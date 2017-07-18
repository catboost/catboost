#include "pcdata.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TPcdata) {
    SIMPLE_UNIT_TEST(TestStress) {
        {
            ui64 key = 0x000017C0B76C4E87ull;
            TString res = EncodeHtmlPcdata(TStringBuf((const char*)&key, sizeof(key)));
        }

        for (size_t i = 0; i < 1000; ++i) {
            const TString s = NUnitTest::RandomString(i, i);

            UNIT_ASSERT_VALUES_EQUAL(DecodeHtmlPcdata(EncodeHtmlPcdata(s)), s);
        }
    }

    SIMPLE_UNIT_TEST(Test1) {
        const TString tests[] = {
            "qw&qw",
            "&<",
            ">&qw",
            "\'&aaa"};

        for (auto s : tests) {
            UNIT_ASSERT_VALUES_EQUAL(DecodeHtmlPcdata(EncodeHtmlPcdata(s)), s);
        }
    }

    SIMPLE_UNIT_TEST(Test2) {
        UNIT_ASSERT_VALUES_EQUAL(EncodeHtmlPcdata("&qqq"), "&amp;qqq");
    }

    SIMPLE_UNIT_TEST(TestEncodeHtmlPcdataAppend) {
        TString s;
        EncodeHtmlPcdataAppend("m&m", s);
        EncodeHtmlPcdataAppend("'s", s);
        UNIT_ASSERT_VALUES_EQUAL(EncodeHtmlPcdata("m&m's"), s);
        UNIT_ASSERT_VALUES_EQUAL("m&amp;m&#39;s", s);
    }

    SIMPLE_UNIT_TEST(TestStrangeAmpParameter) {
        UNIT_ASSERT_VALUES_EQUAL(EncodeHtmlPcdata("m&m's", true), "m&amp;m&#39;s");
        UNIT_ASSERT_VALUES_EQUAL(EncodeHtmlPcdata("m&m's"), "m&amp;m&#39;s"); //default
        UNIT_ASSERT_VALUES_EQUAL(EncodeHtmlPcdata("m&m's", false), "m&m&#39;s");
    }
}
