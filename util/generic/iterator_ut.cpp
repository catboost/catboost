#include "iterator.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TIterator) {
    SIMPLE_UNIT_TEST(ToForwardIteratorTest) {
        yvector<int> x = {1, 2};
        UNIT_ASSERT_VALUES_EQUAL(*std::prev(x.end()), *ToForwardIterator(x.rbegin()));
        UNIT_ASSERT_VALUES_EQUAL(*ToForwardIterator(std::prev(x.rend())), *x.begin());
    }
}
