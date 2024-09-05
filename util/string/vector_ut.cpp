#include <library/cpp/testing/unittest/registar.h>

#include "cast.h"
#include "vector.h"

Y_UNIT_TEST_SUITE(TStringJoinTest) {
    Y_UNIT_TEST(Test1) {
        TVector<TUtf16String> v;

        UNIT_ASSERT_EQUAL(JoinStrings(v, ToWtring("")), ToWtring(""));
    }

    Y_UNIT_TEST(Test2) {
        TVector<TUtf16String> v;

        v.push_back(ToWtring("1"));
        v.push_back(ToWtring("2"));

        UNIT_ASSERT_EQUAL(JoinStrings(v, ToWtring(" ")), ToWtring("1 2"));
    }

    Y_UNIT_TEST(Test3) {
        TVector<TUtf16String> v;

        v.push_back(ToWtring("1"));
        v.push_back(ToWtring("2"));

        UNIT_ASSERT_EQUAL(JoinStrings(v, 1, 10, ToWtring(" ")), ToWtring("2"));
    }

    Y_UNIT_TEST(TestJoinWStrings) {
        const TUtf16String str = u"Яндекс";
        const TVector<TUtf16String> v(1, str);

        UNIT_ASSERT_EQUAL(JoinStrings(v, TUtf16String()), str);
    }
} // Y_UNIT_TEST_SUITE(TStringJoinTest)
