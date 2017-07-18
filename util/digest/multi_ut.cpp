#include "multi.h"

#include <library/unittest/registar.h>

#include <util/stream/output.h>

class TMultiHashTest: public TTestBase {
    UNIT_TEST_SUITE(TMultiHashTest);
    UNIT_TEST(TestStrInt)
    UNIT_TEST(TestIntStr)
    UNIT_TEST(TestSimpleCollision)
    UNIT_TEST_SUITE_END();

private:
    inline void TestStrInt() {
        UNIT_ASSERT_EQUAL(MultiHash(TString("1234567"), static_cast<int>(123)), ULL(17038203285960021630));
    }

    inline void TestIntStr() {
        UNIT_ASSERT_EQUAL(MultiHash(static_cast<int>(123), TString("1234567")), ULL(9973288649881090712));
    }

    inline void TestSimpleCollision() {
        UNIT_ASSERT_UNEQUAL(MultiHash(1, 1, 0), MultiHash(2, 2, 0));
    }
};

UNIT_TEST_SUITE_REGISTRATION(TMultiHashTest);
