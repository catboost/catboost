#include "chartraits.h"

#include <library/cpp/testing/unittest/registar.h>
#include <util/charset/unidata.h>

Y_UNIT_TEST_SUITE(TCharTraits) {
    Y_UNIT_TEST(TestLength) {
        using T = TCharTraits<char>;
        UNIT_ASSERT_EQUAL(T::GetLength("", 0), 0);
        UNIT_ASSERT_EQUAL(T::GetLength("abc", 0), 0);
        UNIT_ASSERT_EQUAL(T::GetLength("abc", 1), 1);
        UNIT_ASSERT_EQUAL(T::GetLength("abc", 3), 3);
        UNIT_ASSERT_EQUAL(T::GetLength("abc", 4), 3);
        UNIT_ASSERT_EQUAL(T::GetLength("abc", 1000), 3);

        // '\0'
        UNIT_ASSERT_EQUAL(T::GetLength("\0", 1000), 0);
        UNIT_ASSERT_EQUAL(T::GetLength("\0abc", 1000), 0);
        UNIT_ASSERT_EQUAL(T::GetLength("a\0bc", 1000), 1);
    }
}
