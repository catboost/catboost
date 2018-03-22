#include "cgiparam.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TCgiParametersTest) {
    SIMPLE_UNIT_TEST(TestScan1) {
        TCgiParameters C;
        C.Scan("aaa=b%62b&ccc=ddd&ag0=");
        UNIT_ASSERT_EQUAL(C.Get("aaa") == "bbb", true);
        UNIT_ASSERT_EQUAL(C.NumOfValues("ag0") == 1, true);
        UNIT_ASSERT(C.Has("ccc", "ddd"));
        UNIT_ASSERT(C.Has("ag0", ""));
        UNIT_ASSERT(!C.Has("a", "bbb"));
        UNIT_ASSERT(!C.Has("aaa", "bb"));

        UNIT_ASSERT(C.Has("ccc"));
        UNIT_ASSERT(!C.Has("zzzzzz"));
    }

    SIMPLE_UNIT_TEST(TestScan2) {
        const TString parsee("=000&aaa=bbb&ag0=&ccc=ddd");
        TCgiParameters c;
        c.Scan(parsee);

        UNIT_ASSERT_VALUES_EQUAL(c.Print(), parsee);
    }

    SIMPLE_UNIT_TEST(TestScan3) {
        const TString parsee("aaa=bbb&ag0=&ccc=ddd");
        TCgiParameters c;
        c.Scan(parsee);

        c.InsertUnescaped("d", "xxx");

        UNIT_ASSERT_VALUES_EQUAL(c.Print(), parsee + "&d=xxx");
    }

    SIMPLE_UNIT_TEST(TestScanAddAll1) {
        TCgiParameters c;
        c.ScanAddAll("qw");

        UNIT_ASSERT_VALUES_EQUAL(c.size(), 1u);
        UNIT_ASSERT(c.Get("qw").empty());
    }

    SIMPLE_UNIT_TEST(TestScanAddAll2) {
        TCgiParameters c;
        c.ScanAddAll("qw&");

        UNIT_ASSERT_VALUES_EQUAL(c.size(), 1u);
        UNIT_ASSERT(c.Get("qw").empty());
    }

    SIMPLE_UNIT_TEST(TestScanAddAll3) {
        TCgiParameters c;
        c.ScanAddAll("qw=1&x");

        UNIT_ASSERT_VALUES_EQUAL(c.size(), 2u);
        UNIT_ASSERT_VALUES_EQUAL(c.Get("qw"), "1");
        UNIT_ASSERT(c.Get("x").empty());
    }

    SIMPLE_UNIT_TEST(TestScanAddAll4) {
        TCgiParameters c;
        c.ScanAddAll("ccc=1&aaa=1&ccc=3&bbb&ccc=2");

        UNIT_ASSERT_VALUES_EQUAL(c.Print(), "aaa=1&bbb=&ccc=1&ccc=3&ccc=2");
    }

    SIMPLE_UNIT_TEST(TestEraseAll) {
        TCgiParameters c;
        c.ScanAddAll("par=1&aaa=1&par=2&bbb&par=3");
        c.EraseAll("par");

        UNIT_ASSERT_VALUES_EQUAL(c.Print(), "aaa=1&bbb=");
    }

    SIMPLE_UNIT_TEST(TestErase) {
        TCgiParameters c;
        c.ScanAddAll("par=1&aaa=1&par=2&bbb&par=3&par=1");

        c.Erase("par", 1);
        UNIT_ASSERT_VALUES_EQUAL(c.Print(), "aaa=1&bbb=&par=1&par=3&par=1");

        c.Erase("par", "1");
        UNIT_ASSERT_VALUES_EQUAL(c.Print(), "aaa=1&bbb=&par=3");
    }

    SIMPLE_UNIT_TEST(TestReplaceUnescaped1) {
        TCgiParameters c;
        c.ScanAddAll("many_keys=1&aaa=1&many_keys=2&bbb&many_keys=3");
        c.ReplaceUnescaped("many_keys", "new_value");

        UNIT_ASSERT_VALUES_EQUAL(c.Print(), "aaa=1&bbb=&many_keys=new_value");
    }

    SIMPLE_UNIT_TEST(TestReplaceUnescaped2) {
        TCgiParameters c;
        c.ScanAddAll("par=1&only_one=1&par=2&bbb&par=3");
        c.ReplaceUnescaped("only_one", "new_value");

        UNIT_ASSERT_VALUES_EQUAL(c.Print(), "bbb=&only_one=new_value&par=1&par=2&par=3");
    }

    SIMPLE_UNIT_TEST(TestReplaceUnescaped3) {
        TCgiParameters c;
        c.ScanAddAll("par=1&aaa=1&par=2&bbb&par=3");
        c.ReplaceUnescaped("no_such_key", "new_value");

        UNIT_ASSERT_VALUES_EQUAL(c.Print(), "aaa=1&bbb=&no_such_key=new_value&par=1&par=2&par=3");
    }

    SIMPLE_UNIT_TEST(TestReplaceUnescapedRange1) {
        TCgiParameters c;
        c.ScanAddAll("par=1&aaa=1&par=2&bbb&par=3");
        c.ReplaceUnescaped("par", {"x", "y", "z"}); // 3 old values, 3 new values

        UNIT_ASSERT_VALUES_EQUAL(c.Print(), "aaa=1&bbb=&par=x&par=y&par=z");
    }

    SIMPLE_UNIT_TEST(TestReplaceUnescapedRange2) {
        TCgiParameters c;
        c.ScanAddAll("par=1&aaa=1&par=2&bbb");
        c.ReplaceUnescaped("par", {"x", "y", "z"}); // 2 old values, 3 new values

        UNIT_ASSERT_VALUES_EQUAL(c.Print(), "aaa=1&bbb=&par=x&par=y&par=z");
    }

    SIMPLE_UNIT_TEST(TestReplaceUnescapedRange3) {
        TCgiParameters c;
        c.ScanAddAll("par=1&aaa=1&par=2&bbb&par=3");
        c.ReplaceUnescaped("par", {"x", "y"}); // 3 old values, 2 new values

        UNIT_ASSERT_VALUES_EQUAL(c.Print(), "aaa=1&bbb=&par=x&par=y");
    }

    SIMPLE_UNIT_TEST(TestNumOfValues) {
        TCgiParameters c;
        c.ScanAddAll("par=1&aaa=1&par=2&bbb&par=3");

        UNIT_ASSERT_VALUES_EQUAL(c.NumOfValues("par"), 3u);
    }

    SIMPLE_UNIT_TEST(TestUnscape) {
        TCgiParameters c("f=1&t=%84R%84%7C%84%80%84%7E&reqenc=SHIFT_JIS&p=0");
        UNIT_ASSERT_VALUES_EQUAL(c.Get("t"), "\x84R\x84\x7C\x84\x80\x84\x7E");
    }

    SIMPLE_UNIT_TEST(TestEmpty) {
        UNIT_ASSERT(TCgiParameters().Print().empty());
    }

    SIMPLE_UNIT_TEST(TestJoinUnescaped) {
        TCgiParameters c;

        c.Scan("foo=1&foo=2");
        c.JoinUnescaped("foo", ';', "0");

        UNIT_ASSERT_VALUES_EQUAL(c.Print(), "foo=1;2;0");
    }

    SIMPLE_UNIT_TEST(TestContInit) {
        TCgiParameters c = {std::make_pair("a", "a1"), std::make_pair("b", "b1"), std::make_pair("a", "a2")};

        UNIT_ASSERT_VALUES_EQUAL(c.NumOfValues("a"), 2u);
        UNIT_ASSERT_VALUES_EQUAL(c.NumOfValues("b"), 1u);

        UNIT_ASSERT_VALUES_EQUAL(c.Get("b"), "b1");
        UNIT_ASSERT_VALUES_EQUAL(c.Get("a", 0), "a1");
        UNIT_ASSERT_VALUES_EQUAL(c.Get("a", 1), "a2");

        UNIT_ASSERT_VALUES_EQUAL(c.Print(), "a=a1&a=a2&b=b1");
    }
}
