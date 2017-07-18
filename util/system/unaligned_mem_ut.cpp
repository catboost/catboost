#include "unaligned_mem.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TUnalignedMem) {
    SIMPLE_UNIT_TEST(TestReadWrite) {
        char buf[100];

        WriteUnaligned(buf, (ui16)1);
        WriteUnaligned(buf + 2, (ui32)2);
        WriteUnaligned(buf + 2 + 4, (ui64)3);

        UNIT_ASSERT_VALUES_EQUAL(ReadUnaligned<ui16>(buf), 1);
        UNIT_ASSERT_VALUES_EQUAL(ReadUnaligned<ui32>(buf + 2), 2);
        UNIT_ASSERT_VALUES_EQUAL(ReadUnaligned<ui64>(buf + 2 + 4), 3);
    }
}
