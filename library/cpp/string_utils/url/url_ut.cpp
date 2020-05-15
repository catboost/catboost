#include "url.h"

#include <util/string/cast.h>

#include <library/cpp/unittest/registar.h>

Y_UNIT_TEST_SUITE(TUtilUrlTest) {
    Y_UNIT_TEST(TestGetHostAndGetHostAndPort) {
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetHost("ya.ru/bebe"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetHostAndPort("ya.ru/bebe"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetHost("ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetHostAndPort("ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetHost("ya.ru:8080"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru:8080", GetHostAndPort("ya.ru:8080"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetHost("ya.ru/bebe:8080"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetHostAndPort("ya.ru/bebe:8080"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetHost("ya.ru:8080/bebe"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetHost("https://ya.ru:8080/bebe"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru:8080", GetHostAndPort("ya.ru:8080/bebe"));
        // irl RFC3986 sometimes gets ignored
        UNIT_ASSERT_VALUES_EQUAL("pravda-kmv.ru", GetHost("pravda-kmv.ru?page=news&id=6973"));
        UNIT_ASSERT_VALUES_EQUAL("pravda-kmv.ru", GetHostAndPort("pravda-kmv.ru?page=news&id=6973"));
    }

    Y_UNIT_TEST(TestGetPathAndQuery) {
        UNIT_ASSERT_VALUES_EQUAL("/", GetPathAndQuery("ru.wikipedia.org"));
        UNIT_ASSERT_VALUES_EQUAL("/", GetPathAndQuery("ru.wikipedia.org/"));
        UNIT_ASSERT_VALUES_EQUAL("/", GetPathAndQuery("ru.wikipedia.org:8080"));
        UNIT_ASSERT_VALUES_EQUAL("/index.php?123/", GetPathAndQuery("ru.wikipedia.org/index.php?123/"));
        UNIT_ASSERT_VALUES_EQUAL("/", GetPathAndQuery("http://ru.wikipedia.org:8080"));
        UNIT_ASSERT_VALUES_EQUAL("/index.php?123/", GetPathAndQuery("https://ru.wikipedia.org/index.php?123/"));
        UNIT_ASSERT_VALUES_EQUAL("/", GetPathAndQuery("ru.wikipedia.org/#comment"));
        UNIT_ASSERT_VALUES_EQUAL("/?1", GetPathAndQuery("ru.wikipedia.org/?1#comment"));
        UNIT_ASSERT_VALUES_EQUAL("/?1#comment", GetPathAndQuery("ru.wikipedia.org/?1#comment", false));
    }

    Y_UNIT_TEST(TestGetDomain) {
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetDomain("www.ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetDomain("ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetDomain("a.b.ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetDomain("ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya", GetDomain("ya"));
        UNIT_ASSERT_VALUES_EQUAL("", GetDomain(""));
    }

    Y_UNIT_TEST(TestGetParentDomain) {
        UNIT_ASSERT_VALUES_EQUAL("", GetParentDomain("www.ya.ru", 0));
        UNIT_ASSERT_VALUES_EQUAL("ru", GetParentDomain("www.ya.ru", 1));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetParentDomain("www.ya.ru", 2));
        UNIT_ASSERT_VALUES_EQUAL("www.ya.ru", GetParentDomain("www.ya.ru", 3));
        UNIT_ASSERT_VALUES_EQUAL("www.ya.ru", GetParentDomain("www.ya.ru", 4));
        UNIT_ASSERT_VALUES_EQUAL("com", GetParentDomain("ya.com", 1));
        UNIT_ASSERT_VALUES_EQUAL("ya.com", GetParentDomain("ya.com", 2));
        UNIT_ASSERT_VALUES_EQUAL("RU", GetParentDomain("RU", 1));
        UNIT_ASSERT_VALUES_EQUAL("RU", GetParentDomain("RU", 2));
        UNIT_ASSERT_VALUES_EQUAL("", GetParentDomain("", 0));
        UNIT_ASSERT_VALUES_EQUAL("", GetParentDomain("", 1));
    }

    Y_UNIT_TEST(TestGetZone) {
        UNIT_ASSERT_VALUES_EQUAL("ru", GetZone("www.ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("com", GetZone("ya.com"));
        UNIT_ASSERT_VALUES_EQUAL("RU", GetZone("RU"));
        UNIT_ASSERT_VALUES_EQUAL("FHFBN", GetZone("ya.FHFBN"));
        UNIT_ASSERT_VALUES_EQUAL("", GetZone(""));
    }

    Y_UNIT_TEST(TestAddSchemePrefix) {
        UNIT_ASSERT_VALUES_EQUAL("http://yandex.ru", AddSchemePrefix("yandex.ru"));
        UNIT_ASSERT_VALUES_EQUAL("http://yandex.ru", AddSchemePrefix("http://yandex.ru"));
        UNIT_ASSERT_VALUES_EQUAL("https://yandex.ru", AddSchemePrefix("https://yandex.ru"));
        UNIT_ASSERT_VALUES_EQUAL("file://yandex.ru", AddSchemePrefix("file://yandex.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ftp://ya.ru", AddSchemePrefix("ya.ru", "ftp"));
    }

    Y_UNIT_TEST(TestSchemeGet) {
        UNIT_ASSERT_VALUES_EQUAL("http://", GetSchemePrefix("http://ya.ru/bebe"));
        UNIT_ASSERT_VALUES_EQUAL("", GetSchemePrefix("yaru"));
        UNIT_ASSERT_VALUES_EQUAL("yaru://", GetSchemePrefix("yaru://ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("", GetSchemePrefix("ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("ftp://", GetSchemePrefix("ftp://ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("https://", GetSchemePrefix("https://")); // is that right?
    }

    Y_UNIT_TEST(TestSchemeCut) {
        UNIT_ASSERT_VALUES_EQUAL("ya.ru/bebe", CutSchemePrefix("http://ya.ru/bebe"));
        UNIT_ASSERT_VALUES_EQUAL("yaru", CutSchemePrefix("yaru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru://zzz", CutSchemePrefix("yaru://ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru://zzz", CutSchemePrefix("ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru://zzz", CutSchemePrefix("ftp://ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("", CutSchemePrefix("https://")); // is that right?

        UNIT_ASSERT_VALUES_EQUAL("ftp://ya.ru", CutHttpPrefix("ftp://ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru/zzz", CutHttpPrefix("http://ya.ru/zzz"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru/zzz", CutHttpPrefix("http://ya.ru/zzz", true));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru/zzz", CutHttpPrefix("https://ya.ru/zzz"));
        UNIT_ASSERT_VALUES_EQUAL("https://ya.ru/zzz", CutHttpPrefix("https://ya.ru/zzz", true));
        UNIT_ASSERT_VALUES_EQUAL("", CutHttpPrefix("https://"));               // is that right?
        UNIT_ASSERT_VALUES_EQUAL("https://", CutHttpPrefix("https://", true)); // is that right?
    }

    Y_UNIT_TEST(TestMisc) {
        UNIT_ASSERT_VALUES_EQUAL("", CutWWWPrefix("www."));
        UNIT_ASSERT_VALUES_EQUAL("", CutWWWPrefix("WwW."));
        UNIT_ASSERT_VALUES_EQUAL("www", CutWWWPrefix("www"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", CutWWWPrefix("www.ya.ru"));

        UNIT_ASSERT_VALUES_EQUAL("", CutWWWNumberedPrefix("www."));
        UNIT_ASSERT_VALUES_EQUAL("www", CutWWWNumberedPrefix("www"));
        UNIT_ASSERT_VALUES_EQUAL("www27", CutWWWNumberedPrefix("www27"));
        UNIT_ASSERT_VALUES_EQUAL("", CutWWWNumberedPrefix("www27."));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", CutWWWNumberedPrefix("www.ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", CutWWWNumberedPrefix("www2.ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", CutWWWNumberedPrefix("www12.ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", CutWWWNumberedPrefix("ww2.ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("w1w2w3.ya.ru", CutWWWNumberedPrefix("w1w2w3.ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("123.ya.ru", CutWWWNumberedPrefix("123.ya.ru"));

        UNIT_ASSERT_VALUES_EQUAL("", CutMPrefix("m."));
        UNIT_ASSERT_VALUES_EQUAL("", CutMPrefix("M."));
        UNIT_ASSERT_VALUES_EQUAL("m", CutMPrefix("m"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", CutMPrefix("m.ya.ru"));
    }

    Y_UNIT_TEST(TestSplitUrlToHostAndPath) {
        TStringBuf host, path;

        SplitUrlToHostAndPath("https://yandex.ru/yandsearch", host, path);
        UNIT_ASSERT_STRINGS_EQUAL(host, "https://yandex.ru");
        UNIT_ASSERT_STRINGS_EQUAL(path, "/yandsearch");

        SplitUrlToHostAndPath("yandex.ru/yandsearch", host, path);
        UNIT_ASSERT_STRINGS_EQUAL(host, "yandex.ru");
        UNIT_ASSERT_STRINGS_EQUAL(path, "/yandsearch");

        SplitUrlToHostAndPath("https://yandex.ru", host, path);
        UNIT_ASSERT_STRINGS_EQUAL(host, "https://yandex.ru");
        UNIT_ASSERT_STRINGS_EQUAL(path, "");

        SplitUrlToHostAndPath("invalid url /", host, path);
        UNIT_ASSERT_STRINGS_EQUAL(host, "invalid url ");
        UNIT_ASSERT_STRINGS_EQUAL(path, "/");
    }

    Y_UNIT_TEST(TestSeparateUrlFromQueryAndFragment) {
        TStringBuf sanitizedUrl, query, fragment;

        SeparateUrlFromQueryAndFragment("https://yandex.ru/yandsearch", sanitizedUrl, query, fragment);
        UNIT_ASSERT_STRINGS_EQUAL(sanitizedUrl, "https://yandex.ru/yandsearch");
        UNIT_ASSERT_STRINGS_EQUAL(query, "");
        UNIT_ASSERT_STRINGS_EQUAL(fragment, "");

        SeparateUrlFromQueryAndFragment("https://yandex.ru/yandsearch?param1=val1&param2=val2", sanitizedUrl, query, fragment);
        UNIT_ASSERT_STRINGS_EQUAL(sanitizedUrl, "https://yandex.ru/yandsearch");
        UNIT_ASSERT_STRINGS_EQUAL(query, "param1=val1&param2=val2");
        UNIT_ASSERT_STRINGS_EQUAL(fragment, "");

        SeparateUrlFromQueryAndFragment("https://yandex.ru/yandsearch#fragment", sanitizedUrl, query, fragment);
        UNIT_ASSERT_STRINGS_EQUAL(sanitizedUrl, "https://yandex.ru/yandsearch");
        UNIT_ASSERT_STRINGS_EQUAL(query, "");
        UNIT_ASSERT_STRINGS_EQUAL(fragment, "fragment");

        SeparateUrlFromQueryAndFragment("https://yandex.ru/yandsearch?param1=val1&param2=val2#fragment", sanitizedUrl, query, fragment);
        UNIT_ASSERT_STRINGS_EQUAL(sanitizedUrl, "https://yandex.ru/yandsearch");
        UNIT_ASSERT_STRINGS_EQUAL(query, "param1=val1&param2=val2");
        UNIT_ASSERT_STRINGS_EQUAL(fragment, "fragment");
    }

    Y_UNIT_TEST(TestGetSchemeHostAndPort) {
        { // all components are present
            TStringBuf scheme("unknown"), host("unknown");
            ui16 port = 0;
            GetSchemeHostAndPort("https://ya.ru:8080/bebe", scheme, host, port);
            UNIT_ASSERT_VALUES_EQUAL(scheme, "https://");
            UNIT_ASSERT_VALUES_EQUAL(host, "ya.ru");
            UNIT_ASSERT_VALUES_EQUAL(port, 8080);
        }
        { // scheme is abset
            TStringBuf scheme("unknown"), host("unknown");
            ui16 port = 0;
            GetSchemeHostAndPort("ya.ru:8080/bebe", scheme, host, port);
            UNIT_ASSERT_VALUES_EQUAL(scheme, "unknown");
            UNIT_ASSERT_VALUES_EQUAL(host, "ya.ru");
            UNIT_ASSERT_VALUES_EQUAL(port, 8080);
        }
        { // scheme and port are absent
            TStringBuf scheme("unknown"), host("unknown");
            ui16 port = 0;
            GetSchemeHostAndPort("ya.ru/bebe", scheme, host, port);
            UNIT_ASSERT_VALUES_EQUAL(scheme, "unknown");
            UNIT_ASSERT_VALUES_EQUAL(host, "ya.ru");
            UNIT_ASSERT_VALUES_EQUAL(port, 0);
        }
        { // port is absent, but returned its default value for HTTP
            TStringBuf scheme("unknown"), host("unknown");
            ui16 port = 0;
            GetSchemeHostAndPort("http://ya.ru/bebe", scheme, host, port);
            UNIT_ASSERT_VALUES_EQUAL(scheme, "http://");
            UNIT_ASSERT_VALUES_EQUAL(host, "ya.ru");
            UNIT_ASSERT_VALUES_EQUAL(port, 80);
        }
        { // port is absent, but returned its default value for HTTPS
            TStringBuf scheme("unknown"), host("unknown");
            ui16 port = 0;
            GetSchemeHostAndPort("https://ya.ru/bebe", scheme, host, port);
            UNIT_ASSERT_VALUES_EQUAL(scheme, "https://");
            UNIT_ASSERT_VALUES_EQUAL(host, "ya.ru");
            UNIT_ASSERT_VALUES_EQUAL(port, 443);
        }
        { // ipv6
            TStringBuf scheme("unknown"), host("unknown");
            ui16 port = 0;
            GetSchemeHostAndPort("https://[1080:0:0:0:8:800:200C:417A]:443/bebe", scheme, host, port);
            UNIT_ASSERT_VALUES_EQUAL(scheme, "https://");
            UNIT_ASSERT_VALUES_EQUAL(host, "[1080:0:0:0:8:800:200C:417A]");
            UNIT_ASSERT_VALUES_EQUAL(port, 443);
        }
        { // ipv6
            TStringBuf scheme("unknown"), host("unknown");
            ui16 port = 0;
            GetSchemeHostAndPort("[::1]/bebe", scheme, host, port);
            UNIT_ASSERT_VALUES_EQUAL(scheme, "unknown");
            UNIT_ASSERT_VALUES_EQUAL(host, "[::1]");
            UNIT_ASSERT_VALUES_EQUAL(port, 0);
        }
        { // ipv6
            TStringBuf scheme("unknown"), host("unknown");
            ui16 port = 0;
            GetSchemeHostAndPort("unknown:///bebe", scheme, host, port);
            UNIT_ASSERT_VALUES_EQUAL(scheme, "unknown://");
            UNIT_ASSERT_VALUES_EQUAL(host, "");
            UNIT_ASSERT_VALUES_EQUAL(port, 0);
        }
        // port overflow
        auto testCase = []() {
            TStringBuf scheme("unknown"), host("unknown");
            ui16 port = 0;
            GetSchemeHostAndPort("https://ya.ru:65536/bebe", scheme, host, port);
        };
        UNIT_ASSERT_EXCEPTION(testCase(), yexception);
    }

    Y_UNIT_TEST(TestCutUrlPrefixes) {
        UNIT_ASSERT_VALUES_EQUAL("ya.ru/bebe", CutUrlPrefixes("http://ya.ru/bebe"));
        UNIT_ASSERT_VALUES_EQUAL("yaru", CutUrlPrefixes("yaru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru://zzz", CutUrlPrefixes("yaru://ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru://zzz", CutUrlPrefixes("ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru://zzz", CutUrlPrefixes("ftp://ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("", CutUrlPrefixes("https://"));

        UNIT_ASSERT_VALUES_EQUAL("ya.ru/bebe", CutUrlPrefixes("https://www.ya.ru/bebe"));
        UNIT_ASSERT_VALUES_EQUAL("yaru", CutUrlPrefixes("www.yaru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru://zzz", CutUrlPrefixes("yaru://www.ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru://zzz", CutUrlPrefixes("www.ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru://zzz", CutUrlPrefixes("ftp://www.ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("", CutUrlPrefixes("http://www."));
    }

    Y_UNIT_TEST(TestUrlPathStartWithToken) {
        UNIT_ASSERT_VALUES_EQUAL(true, DoesUrlPathStartWithToken("http://ya.ru/bebe/zzz", "bebe"));
        UNIT_ASSERT_VALUES_EQUAL(true, DoesUrlPathStartWithToken("http://ya.ru/bebe?zzz", "bebe"));
        UNIT_ASSERT_VALUES_EQUAL(true, DoesUrlPathStartWithToken("http://ya.ru/bebe/", "bebe"));
        UNIT_ASSERT_VALUES_EQUAL(true, DoesUrlPathStartWithToken("http://ya.ru/bebe?", "bebe"));
        UNIT_ASSERT_VALUES_EQUAL(true, DoesUrlPathStartWithToken("https://ya.ru/bebe", "bebe"));
        UNIT_ASSERT_VALUES_EQUAL(false, DoesUrlPathStartWithToken("http://ya.ru/bebe.zzz", "bebe"));
        UNIT_ASSERT_VALUES_EQUAL(false, DoesUrlPathStartWithToken("http://ya.ru/", "bebe"));
        UNIT_ASSERT_VALUES_EQUAL(false, DoesUrlPathStartWithToken("http://ya.ru", "bebe"));
        UNIT_ASSERT_VALUES_EQUAL(false, DoesUrlPathStartWithToken("http://bebe", "bebe"));
        UNIT_ASSERT_VALUES_EQUAL(false, DoesUrlPathStartWithToken("https://bebe/", "bebe"));
    }
}
