#include "util.h"

#include <library/cpp/testing/unittest/registar.h>

class TStrUtilTest: public TTestBase {
    UNIT_TEST_SUITE(TStrUtilTest);
    UNIT_TEST(TestSpn);
    UNIT_TEST(TestRemoveAll);
    UNIT_TEST_SUITE_END();

public:
    void TestSpn() {
        str_spn rul("a-z", true);
        char s[] = "!@#$ab%^&c+-";
        UNIT_ASSERT_EQUAL(rul.brk(s), s + 4);
        UNIT_ASSERT_EQUAL(rul.brk(s + 4), s + 4);
        UNIT_ASSERT_EQUAL(rul.brk(s + 10), s + 12);
        char* s1 = s;
        UNIT_ASSERT_EQUAL(strcmp(rul.sep(s1), "!@#$"), 0);
        UNIT_ASSERT_EQUAL(strcmp(rul.sep(s1), ""), 0);
        UNIT_ASSERT_EQUAL(strcmp(rul.sep(s1), "%^&"), 0);
        UNIT_ASSERT_EQUAL(strcmp(rul.sep(s1), "+-"), 0);
        UNIT_ASSERT_EQUAL(rul.sep(s1), nullptr);
    }

    void TestRemoveAll() {
        static const struct T {
            const char* Str;
            char Ch;
            const char* Result;
        } tests[] = {
            {"", 'x', ""},
            {"hello world", 'h', "ello world"},
            {"hello world", 'l', "heo word"},
            {"hello world", 'x', "hello world"},
        };

        for (const T* t = tests; t != std::end(tests); ++t) {
            TString str(t->Str);
            RemoveAll(str, t->Ch);
            UNIT_ASSERT_EQUAL(t->Result, str);
        }
    }
};

UNIT_TEST_SUITE_REGISTRATION(TStrUtilTest);
