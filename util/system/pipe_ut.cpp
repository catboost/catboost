#include "pipe.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TPipeTest) {
    SIMPLE_UNIT_TEST(TestPipe) {
        TPipe r;
        TPipe w;
        TPipe::Pipe(r, w);
        char c = 'a';
        UNIT_ASSERT(1 == w.Write(&c, 1));
        UNIT_ASSERT(1 == r.Read(&c, 1));
        UNIT_ASSERT_VALUES_EQUAL('a', c);
    }
}
