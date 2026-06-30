#include "shmat.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TTestSharedMemory) {
    Y_UNIT_TEST(TestInProc) {
        TSharedMemory m1;
        TSharedMemory m2;

        UNIT_ASSERT(m1.Create(128));
        UNIT_ASSERT(m2.Open(m1.GetId(), m1.GetSize()));

        *(ui32*)m1.GetPtr() = 123;

        UNIT_ASSERT_VALUES_EQUAL(*(ui32*)m2.GetPtr(), 123);
    }
} // Y_UNIT_TEST_SUITE(TTestSharedMemory)
