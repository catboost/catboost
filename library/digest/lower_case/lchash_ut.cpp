#include "lchash.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TWebDaemonHash) {
    SIMPLE_UNIT_TEST(Stability) {
        UNIT_ASSERT_VALUES_EQUAL(FnvCaseLess<ui64>("blah"), ULL(5923727754379976229));
        UNIT_ASSERT_VALUES_EQUAL(FnvCaseLess<ui64>("blahminor"), ULL(8755704309003440816));
    }

    SIMPLE_UNIT_TEST(CaseLess) {
        UNIT_ASSERT_VALUES_EQUAL(FnvCaseLess<ui64>("blah"), FnvCaseLess<ui64>("bLah"));
        UNIT_ASSERT_VALUES_EQUAL(FnvCaseLess<ui64>("blah"), FnvCaseLess<ui64>("blAh"));
        UNIT_ASSERT_VALUES_EQUAL(FnvCaseLess<ui64>("blah"), FnvCaseLess<ui64>("BLAH"));
    }

    SIMPLE_UNIT_TEST(Robustness) {
        UNIT_ASSERT(FnvCaseLess<ui64>("x-real-ip") != FnvCaseLess<ui64>("x-req-id"));
        UNIT_ASSERT(FnvCaseLess<ui64>("x-real-ip") != FnvCaseLess<ui64>("x-start-time"));
    }
}
