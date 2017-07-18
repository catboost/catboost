#include "url.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TUtilUrlTest) {
    SIMPLE_UNIT_TEST(TestGetHostAndGetHostAndPort) {
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

    SIMPLE_UNIT_TEST(TestGetPathAndQuery) {
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

    SIMPLE_UNIT_TEST(TestGetDomain) {
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetDomain("www.ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetDomain("ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetDomain("a.b.ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", GetDomain("ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ya", GetDomain("ya"));
        UNIT_ASSERT_VALUES_EQUAL("", GetDomain(""));
    }

    SIMPLE_UNIT_TEST(TestGetParentDomain) {
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

    SIMPLE_UNIT_TEST(TestGetZone) {
        UNIT_ASSERT_VALUES_EQUAL("ru", GetZone("www.ya.ru"));
        UNIT_ASSERT_VALUES_EQUAL("com", GetZone("ya.com"));
        UNIT_ASSERT_VALUES_EQUAL("RU", GetZone("RU"));
        UNIT_ASSERT_VALUES_EQUAL("FHFBN", GetZone("ya.FHFBN"));
        UNIT_ASSERT_VALUES_EQUAL("", GetZone(""));
    }

    SIMPLE_UNIT_TEST(TestAddSchemePrefix) {
        UNIT_ASSERT_VALUES_EQUAL("http://yandex.ru", AddSchemePrefix("yandex.ru"));
        UNIT_ASSERT_VALUES_EQUAL("http://yandex.ru", AddSchemePrefix("http://yandex.ru"));
        UNIT_ASSERT_VALUES_EQUAL("https://yandex.ru", AddSchemePrefix("https://yandex.ru"));
        UNIT_ASSERT_VALUES_EQUAL("file://yandex.ru", AddSchemePrefix("file://yandex.ru"));
        UNIT_ASSERT_VALUES_EQUAL("ftp://ya.ru", AddSchemePrefix("ya.ru", "ftp"));
    }

    SIMPLE_UNIT_TEST(TestSchemeGet) {
        UNIT_ASSERT_VALUES_EQUAL("http://", GetSchemePrefix("http://ya.ru/bebe"));
        UNIT_ASSERT_VALUES_EQUAL("", GetSchemePrefix("yaru"));
        UNIT_ASSERT_VALUES_EQUAL("yaru://", GetSchemePrefix("yaru://ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("", GetSchemePrefix("ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("ftp://", GetSchemePrefix("ftp://ya.ru://zzz"));
        UNIT_ASSERT_VALUES_EQUAL("https://", GetSchemePrefix("https://")); // is that right?
    }

    SIMPLE_UNIT_TEST(TestSchemeCut) {
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

    SIMPLE_UNIT_TEST(TestMisc) {
        UNIT_ASSERT_VALUES_EQUAL("", CutWWWPrefix("www."));
        UNIT_ASSERT_VALUES_EQUAL("", CutWWWPrefix("WwW."));
        UNIT_ASSERT_VALUES_EQUAL("www", CutWWWPrefix("www"));
        UNIT_ASSERT_VALUES_EQUAL("ya.ru", CutWWWPrefix("www.ya.ru"));
    }
}
