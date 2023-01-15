#include "hash_ops.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestCIHash) {
    Y_UNIT_TEST(TestYHash1) {
        THashMap<TStringBuf, int, TCIOps, TCIOps> h;

        h["Ab"] = 1;
        h["aB"] = 2;

        UNIT_ASSERT_VALUES_EQUAL(h.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(h["ab"], 2);
    }

    Y_UNIT_TEST(TestYHash2) {
        THashMap<const char*, int, TCIOps, TCIOps> h;

        h["Ab"] = 1;
        h["aB"] = 2;

        UNIT_ASSERT_VALUES_EQUAL(h.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(h["ab"], 2);

        h["Bc"] = 2;
        h["bC"] = 3;

        UNIT_ASSERT_VALUES_EQUAL(h.size(), 2);
        UNIT_ASSERT_VALUES_EQUAL(h["bc"], 3);
    }

    Y_UNIT_TEST(Test1) {
        UNIT_ASSERT_VALUES_EQUAL(TCIOps()("aBc3"), TCIOps()(TStringBuf("AbC3")));
        UNIT_ASSERT(TCIOps()("aBc4", "AbC4"));
    }
}
