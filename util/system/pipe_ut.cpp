#include "pipe.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TPipeTest) {
    Y_UNIT_TEST(TestPipe) {
        TPipe r;
        TPipe w;
        TPipe::Pipe(r, w);
        char c = 'a';
        UNIT_ASSERT(1 == w.Write(&c, 1));
        UNIT_ASSERT(1 == r.Read(&c, 1));
        UNIT_ASSERT_VALUES_EQUAL('a', c);
    }
} // Y_UNIT_TEST_SUITE(TPipeTest)
