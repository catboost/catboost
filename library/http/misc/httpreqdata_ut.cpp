#include "httpreqdata.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TRequestServerDataTest) {
    SIMPLE_UNIT_TEST(Headers) {
        TServerRequestData sd;

        sd.AddHeader("x-xx", "y-yy");
        sd.AddHeader("x-Xx", "y-yy");

        UNIT_ASSERT_VALUES_EQUAL(sd.HeadersCount(), 1);

        sd.AddHeader("x-XxX", "y-yyy");
        UNIT_ASSERT_VALUES_EQUAL(sd.HeadersCount(), 2);
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(sd.HeaderIn("X-XX")), AsStringBuf("y-yy"));
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(sd.HeaderIn("X-XXX")), AsStringBuf("y-yyy"));
    }

    SIMPLE_UNIT_TEST(ComplexHeaders) {
        TServerRequestData sd;
        sd.SetHost("zzz", 1);

        sd.AddHeader("x-Xx", "y-yy");
        UNIT_ASSERT_VALUES_EQUAL(sd.HeadersCount(), 1);
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(sd.HeaderIn("X-XX")), AsStringBuf("y-yy"));

        sd.AddHeader("x-Xz", "y-yy");
        UNIT_ASSERT_VALUES_EQUAL(sd.HeadersCount(), 2);
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(sd.HeaderIn("X-Xz")), AsStringBuf("y-yy"));

        UNIT_ASSERT_VALUES_EQUAL(sd.ServerName(), "zzz");
        UNIT_ASSERT_VALUES_EQUAL(sd.ServerPort(), "1");
        sd.AddHeader("Host", "1234");
        UNIT_ASSERT_VALUES_EQUAL(sd.HeadersCount(), 3);
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(sd.HeaderIn("Host")), AsStringBuf("1234"));
        UNIT_ASSERT_VALUES_EQUAL(sd.ServerName(), "1234");
        sd.AddHeader("Host", "12345:678");
        UNIT_ASSERT_VALUES_EQUAL(sd.HeadersCount(), 3);
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(sd.HeaderIn("Host")), AsStringBuf("12345:678"));
        UNIT_ASSERT_VALUES_EQUAL(sd.ServerName(), "12345");
        UNIT_ASSERT_VALUES_EQUAL(sd.ServerPort(), "678");
    }

    SIMPLE_UNIT_TEST(ParseScan) {
        TServerRequestData rd;

        // Parse parses url without host
        UNIT_ASSERT(!rd.Parse(" http://yandex.ru/yandsearch?&gta=fake&haha=da HTTP 1.1 OK"));

        // This should work
        UNIT_ASSERT(rd.Parse(" /yandsearch?&gta=fake&haha=da HTTP 1.1 OK"));

        rd.Scan();
        UNIT_ASSERT(rd.CgiParam.Has("gta", "fake"));
        UNIT_ASSERT(rd.CgiParam.Has("haha", "da"));
        UNIT_ASSERT(!rd.CgiParam.Has("no-param"));

        rd.Clear();
    }

    SIMPLE_UNIT_TEST(Ctor) {
        TServerRequestData rd(" /yandsearch?&gta=fake&haha=da");
        UNIT_ASSERT(rd.CgiParam.Has("gta"));
        UNIT_ASSERT(rd.CgiParam.Has("haha"));
        UNIT_ASSERT(!rd.CgiParam.Has("no-param"));
    }

    SIMPLE_UNIT_TEST(HashCut) {
        TServerRequestData rd;
        rd.Parse(" /yandsearch?&gta=fake&haha=da#&uberParam=yes&q=? HTTP 1.1 OK");

        rd.Scan();
        UNIT_ASSERT(rd.CgiParam.Has("gta"));
        UNIT_ASSERT(rd.CgiParam.Has("haha"));
        UNIT_ASSERT(!rd.CgiParam.Has("uberParam"));
    }

    SIMPLE_UNIT_TEST(MisplacedHashCut) {
        TServerRequestData rd;
        rd.Parse(" /y#ndsearch?&gta=fake&haha=da&uberParam=yes&q=? HTTP 1.1 OK");

        rd.Scan();
        UNIT_ASSERT(rd.CgiParam.empty());
    }

    SIMPLE_UNIT_TEST(CornerCase) {
        TServerRequestData rd;
        rd.Parse(" /yandsearch?#");

        rd.Scan();
        UNIT_ASSERT(rd.CgiParam.empty());
    }

    SIMPLE_UNIT_TEST(AppendQueryString) {
        TServerRequestData rd(" /yandsearch?&gta=fake&haha=da");

        UNIT_ASSERT(rd.CgiParam.Has("gta", "fake"));
        UNIT_ASSERT(rd.CgiParam.Has("haha", "da"));

        const TStringBuf appendix = AsStringBuf("&gta=true&gta=new");
        rd.AppendQueryString(~appendix, +appendix);
        rd.Scan();

        UNIT_ASSERT(rd.CgiParam.Has("gta", "true"));
        UNIT_ASSERT(rd.CgiParam.Has("gta", "new"));
    }

    SIMPLE_UNIT_TEST(SetRemoteAddrSimple) {
        static const TString TEST = "abacaba.search.yandex.net";

        TServerRequestData rd;
        rd.SetRemoteAddr(TEST);
        UNIT_ASSERT_STRINGS_EQUAL(TEST, rd.RemoteAddr());
    }

    SIMPLE_UNIT_TEST(SetRemoteAddrRandom) {
        for (size_t size = 0; size < 2 * INET6_ADDRSTRLEN; ++size) {
            const TString test = NUnitTest::RandomString(size, size);
            TServerRequestData rd;
            rd.SetRemoteAddr(test);
            UNIT_ASSERT_STRINGS_EQUAL(test.substr(0, INET6_ADDRSTRLEN - 1), rd.RemoteAddr());
        }
    }

} // TRequestServerDataTest
