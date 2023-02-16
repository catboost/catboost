#include "quote.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TCGIEscapeTest) {
    Y_UNIT_TEST(ReturnsEndOfTo) {
        char r[10];
        const char* returned = CGIEscape(r, "123");
        UNIT_ASSERT_VALUES_EQUAL(r + strlen("123"), returned);
        UNIT_ASSERT_VALUES_EQUAL('\0', *returned);
    }

    Y_UNIT_TEST(NotZeroTerminated) {
        char r[] = {'1', '2', '3', '4'};
        char buf[sizeof(r) * 3 + 2];

        TString ret(buf, CGIEscape(buf, r, sizeof(r)));

        UNIT_ASSERT_EQUAL(ret, "1234");
    }

    Y_UNIT_TEST(StringBuf) {
        char tmp[100];

        UNIT_ASSERT_VALUES_EQUAL(CgiEscape(tmp, "!@#$%^&*(){}[]\" "), TStringBuf("!@%23$%25%5E%26*%28%29%7B%7D%5B%5D%22+"));
    }

    Y_UNIT_TEST(StrokaRet) {
        UNIT_ASSERT_VALUES_EQUAL(CGIEscapeRet("!@#$%^&*(){}[]\" "), TString("!@%23$%25%5E%26*%28%29%7B%7D%5B%5D%22+"));
    }

    Y_UNIT_TEST(StrokaAppendRet) {
        TString param;
        AppendCgiEscaped("!@#$%^&*(){}[]\" ", param);
        UNIT_ASSERT_VALUES_EQUAL(param, TString("!@%23$%25%5E%26*%28%29%7B%7D%5B%5D%22+"));

        TString param2 = "&param=";
        AppendCgiEscaped("!@#$%^&*(){}[]\" ", param2);
        UNIT_ASSERT_VALUES_EQUAL(param2,
            TString("&param=!@%23$%25%5E%26*%28%29%7B%7D%5B%5D%22+"));

        param2.append("&param_param=");
        AppendCgiEscaped("!@#$%^&*(){}[]\" ", param2);
        UNIT_ASSERT_VALUES_EQUAL(param2,
            TString("&param=!@%23$%25%5E%26*%28%29%7B%7D%5B%5D%22+&param_param=!@%23$%25%5E%26*%28%29%7B%7D%5B%5D%22+"));
    }

}

Y_UNIT_TEST_SUITE(TCGIUnescapeTest) {
    Y_UNIT_TEST(StringBuf) {
        char tmp[100];

        UNIT_ASSERT_VALUES_EQUAL(CgiUnescape(tmp, "!@%23$%25^%26*%28%29"), TStringBuf("!@#$%^&*()"));
    }

    Y_UNIT_TEST(TestValidZeroTerm) {
        char r[10];

        CGIUnescape(r, "1234");
        UNIT_ASSERT_VALUES_EQUAL(r, "1234");

        CGIUnescape(r, "%3d");
        UNIT_ASSERT_VALUES_EQUAL(r, "=");

        CGIUnescape(r, "12%3D34");
        UNIT_ASSERT_VALUES_EQUAL(r, "12=34");
    }

    Y_UNIT_TEST(TestInvalidZeroTerm) {
        char r[10];

        CGIUnescape(r, "%");
        UNIT_ASSERT_VALUES_EQUAL(r, "%");

        CGIUnescape(r, "%3");
        UNIT_ASSERT_VALUES_EQUAL(r, "%3");

        CGIUnescape(r, "%3g");
        UNIT_ASSERT_VALUES_EQUAL(r, "%3g");

        CGIUnescape(r, "12%3g34");
        UNIT_ASSERT_VALUES_EQUAL(r, "12%3g34");

        CGIUnescape(r, "%3u123");
        UNIT_ASSERT_VALUES_EQUAL(r, "%3u123");
    }

    Y_UNIT_TEST(TestValidNotZeroTerm) {
        char r[10];

        CGIUnescape(r, "123456789", 4);
        UNIT_ASSERT_VALUES_EQUAL(r, "1234");

        CGIUnescape(r, "%3d1234", 3);
        UNIT_ASSERT_VALUES_EQUAL(r, "=");

        CGIUnescape(r, "12%3D345678", 7);
        UNIT_ASSERT_VALUES_EQUAL(r, "12=34");
    }

    Y_UNIT_TEST(TestInvalidNotZeroTerm) {
        char r[10];

        CGIUnescape(r, "%3d", 1);
        UNIT_ASSERT_VALUES_EQUAL(r, "%");

        CGIUnescape(r, "%3d", 2);
        UNIT_ASSERT_VALUES_EQUAL(r, "%3");

        CGIUnescape(r, "%3g1234", 3);
        UNIT_ASSERT_VALUES_EQUAL(r, "%3g");

        CGIUnescape(r, "12%3g345678", 7);
        UNIT_ASSERT_VALUES_EQUAL(r, "12%3g34");

        CGIUnescape(r, "%3u1234", 2);
        UNIT_ASSERT_VALUES_EQUAL(r, "%3");

        CGIUnescape(r, "%3u1234", 3);
        UNIT_ASSERT_VALUES_EQUAL(r, "%3u");

        CGIUnescape(r, "%3u1234", 4);
        UNIT_ASSERT_VALUES_EQUAL(r, "%3u1");
    }

    Y_UNIT_TEST(StrokaOutParameterInplace) {
        TString s;

        s = "hello%3dworld";
        CGIUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "hello=world");

        s = "+%23+";
        CGIUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, " # ");

        s = "hello%3u";
        CGIUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "hello%3u");

        s = "0123456789012345";
        CGIUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "0123456789012345");

        s = "";
        CGIUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "");
    }

    Y_UNIT_TEST(StrokaOutParameterNotInplace) {
        TString s, sCopy;

        s = "hello%3dworld";
        sCopy = s;
        CGIUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "hello=world");

        s = "+%23+";
        sCopy = s;
        CGIUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, " # ");

        s = "hello%3u";
        sCopy = s;
        CGIUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "hello%3u");

        s = "0123456789012345";
        sCopy = s;
        CGIUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "0123456789012345");

        s = "";
        sCopy = s;
        CGIUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "");
    }
}

Y_UNIT_TEST_SUITE(TUrlEscapeTest) {
    Y_UNIT_TEST(EscapeEscaped) {
        TString s;

        s = "hello%3dworld";
        UNIT_ASSERT_VALUES_EQUAL(UrlEscapeRet(s), "hello%3dworld");
        UrlEscape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "hello%3dworld");
    }

    Y_UNIT_TEST(EscapeUnescape) {
        TString s;

        s = "hello%3dworld";
        UrlEscape(s);
        UrlUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "hello=world");
    }

    Y_UNIT_TEST(EscapeUnescapeRet) {
        TString s;

        s = "hello%3dworld";
        UNIT_ASSERT_VALUES_EQUAL(UrlUnescapeRet(UrlEscapeRet(s)), "hello=world");
    }

    Y_UNIT_TEST(EscapeEscapedForce) {
        TString s;

        s = "hello%3dworld";
        UNIT_ASSERT_VALUES_EQUAL(UrlEscapeRet(s, true), "hello%253dworld");
        UrlEscape(s, true);
        UNIT_ASSERT_VALUES_EQUAL(s, "hello%253dworld");
    }

    Y_UNIT_TEST(EscapeUnescapeForce) {
        TString s;

        s = "hello%3dworld";
        UrlEscape(s, true);
        UrlUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "hello%3dworld");
    }

    Y_UNIT_TEST(EscapeUnescapeForceRet) {
        TString s;

        s = "hello%3dworld";
        UNIT_ASSERT_VALUES_EQUAL(UrlUnescapeRet(UrlEscapeRet(s, true)), "hello%3dworld");
    }
}

Y_UNIT_TEST_SUITE(TUrlUnescapeTest) {
    Y_UNIT_TEST(StrokaOutParameterInplace) {
        TString s;

        s = "hello%3dworld";
        UrlUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "hello=world");

        s = "+%23+";
        UrlUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "+#+");

        s = "hello%3u";
        UrlUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "hello%3u");

        s = "0123456789012345";
        UrlUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "0123456789012345");

        s = "";
        UrlUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "");
    }

    Y_UNIT_TEST(StrokaOutParameterNotInplace) {
        TString s, sCopy;

        s = "hello%3dworld";
        sCopy = s;
        UrlUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "hello=world");

        s = "+%23+";
        sCopy = s;
        UrlUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "+#+");

        s = "hello%3u";
        sCopy = s;
        UrlUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "hello%3u");

        s = "0123456789012345";
        sCopy = s;
        UrlUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "0123456789012345");

        s = "";
        sCopy = s;
        UrlUnescape(s);
        UNIT_ASSERT_VALUES_EQUAL(s, "");
    }
}

Y_UNIT_TEST_SUITE(TQuoteTest) {
    Y_UNIT_TEST(ReturnsEndOfTo) {
        char r[10];
        const char* returned = Quote(r, "123");
        UNIT_ASSERT_VALUES_EQUAL(r + strlen("123"), returned);
        UNIT_ASSERT_VALUES_EQUAL('\0', *returned);
    }

    Y_UNIT_TEST(SlashIsSafeByDefault) {
        char r[100];
        Quote(r, "/path;tail/path,tail/");
        UNIT_ASSERT_VALUES_EQUAL("/path%3Btail/path%2Ctail/", r);
        TString s("/path;tail/path,tail/");
        Quote(s);
        UNIT_ASSERT_VALUES_EQUAL("/path%3Btail/path%2Ctail/", s.c_str());
    }

    Y_UNIT_TEST(SafeColons) {
        char r[100];
        Quote(r, "/path;tail/path,tail/", ";,");
        UNIT_ASSERT_VALUES_EQUAL("%2Fpath;tail%2Fpath,tail%2F", r);
        TString s("/path;tail/path,tail/");
        Quote(s, ";,");
        UNIT_ASSERT_VALUES_EQUAL("%2Fpath;tail%2Fpath,tail%2F", s.c_str());
    }

    Y_UNIT_TEST(StringBuf) {
        char r[100];
        char* end = Quote(r, "abc\0/path", "");
        UNIT_ASSERT_VALUES_EQUAL("abc\0%2Fpath", TStringBuf(r, end));
    }
}
