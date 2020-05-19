#include "date.h"

#include <library/cpp/unittest/registar.h>

Y_UNIT_TEST_SUITE(TDateTest) {
    Y_UNIT_TEST(ComponentsTest) {
        {
            TDate d("20110215");
            UNIT_ASSERT_EQUAL(d.GetYear(), 2011);
            UNIT_ASSERT_EQUAL(d.GetMonth(), 2);
            UNIT_ASSERT_EQUAL(d.GetMonthDay(), 15);
            UNIT_ASSERT_EQUAL(d.ToStroka("%Y%m%d"), "20110215");
            UNIT_ASSERT_EQUAL(d.ToStroka(), "20110215");
            UNIT_ASSERT_EQUAL(d.ToStroka("%Y--%m--%d"), "2011--02--15");
            UNIT_ASSERT_EQUAL(d.ToStroka("%U"), "07");
        }
        {
            TDate d(2005, 6, 3);
            UNIT_ASSERT_EQUAL(d.GetYear(), 2005);
            UNIT_ASSERT_EQUAL(d.GetMonth(), 6);
            UNIT_ASSERT_EQUAL(d.GetMonthDay(), 3);
            UNIT_ASSERT_EQUAL(d.ToStroka(), "20050603");
            UNIT_ASSERT_EQUAL(d.ToStroka("____%Y__%m____%d"), "____2005__06____03");
        }
        {
            TDate d("2011-02-15", "%Y-%m-%d");
            UNIT_ASSERT_EQUAL(d.GetYear(), 2011);
            UNIT_ASSERT_EQUAL(d.GetMonth(), 2);
            UNIT_ASSERT_EQUAL(d.GetMonthDay(), 15);
            UNIT_ASSERT_EQUAL(d.ToStroka("%Y%m%d"), "20110215");
        }
    }
}
