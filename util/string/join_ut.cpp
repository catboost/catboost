#include "join.h"

#include <library/unittest/registar.h>
#include <util/generic/vector.h>

#include <util/stream/output.h>

struct TCustomData {
    yvector<int> Ints;
};

TString ToString(const TCustomData& d) {
    return JoinSeq("__", d.Ints);
}

SIMPLE_UNIT_TEST_SUITE(JoinStringTest) {
    SIMPLE_UNIT_TEST(ScalarItems) {
        UNIT_ASSERT_EQUAL(Join(',', 10, 11.1, "foobar"), "10,11.1,foobar");
        UNIT_ASSERT_EQUAL(Join(", ", 10, 11.1, "foobar"), "10, 11.1, foobar");
        UNIT_ASSERT_EQUAL(Join(", ", 10, 11.1, TString("foobar")), "10, 11.1, foobar");

        UNIT_ASSERT_EQUAL(Join('#', 0, "a", "foobar", -1.4, STRINGBUF("aaa")), "0#a#foobar#-1.4#aaa");
        UNIT_ASSERT_EQUAL(Join("", "", ""), "");
        UNIT_ASSERT_EQUAL(Join("", "a", "b", "c"), "abc");
        UNIT_ASSERT_EQUAL(Join("", "a", "b", "", "c"), "abc");
        UNIT_ASSERT_EQUAL(Join(" ", "a", "b", "", "c"), "a b  c");
    }

    SIMPLE_UNIT_TEST(ContainerItems) {
        int v[] = {1, 2, 3};
        yvector<int> vv(v, v + 3);
        UNIT_ASSERT_EQUAL(JoinSeq(" ", vv), "1 2 3");
        UNIT_ASSERT_EQUAL(JoinSeq(" ", vv), JoinRange(" ", vv.begin(), vv.end()));
        UNIT_ASSERT_EQUAL(JoinRange(" ", v, v + 2), "1 2");
        UNIT_ASSERT_EQUAL(JoinSeq(" ", {1, 2, 3}), "1 2 3");
    }

    SIMPLE_UNIT_TEST(CustomToString) {
        TCustomData d1{{1, 2, 3, 4, 5}};
        TCustomData d2{{0, -1, -2}};
        UNIT_ASSERT_EQUAL(Join(" ", d1, d2), "1__2__3__4__5 0__-1__-2");
    }

    SIMPLE_UNIT_TEST(JoinChars) {
        // Note that char delimeter is printed as single char string,
        // but joined char values are printed as their numeric codes! O_o
        UNIT_ASSERT_EQUAL(Join('a', 'a', 'a'), "97a97");
        UNIT_ASSERT_EQUAL(Join("a", "a", "a"), "aaa");
    }
}
