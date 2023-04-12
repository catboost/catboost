#include "platform.h"

#include <library/cpp/testing/unittest/registar.h>

class TPlatformTest: public TTestBase {
    UNIT_TEST_SUITE(TPlatformTest);
    UNIT_TEST(TestSizeOf)
    UNIT_TEST_SUITE_END();

private:
    inline void TestSizeOf() {
        UNIT_ASSERT_EQUAL(SIZEOF_PTR, sizeof(void*));
        UNIT_ASSERT_EQUAL(SIZEOF_CHAR, sizeof(char));
        UNIT_ASSERT_EQUAL(SIZEOF_SHORT, sizeof(short));
        UNIT_ASSERT_EQUAL(SIZEOF_INT, sizeof(int));
        UNIT_ASSERT_EQUAL(SIZEOF_LONG, sizeof(long));
        UNIT_ASSERT_EQUAL(SIZEOF_LONG_LONG, sizeof(long long));
        UNIT_ASSERT_EQUAL(SIZEOF_UNSIGNED_CHAR, sizeof(unsigned char));
        UNIT_ASSERT_EQUAL(SIZEOF_UNSIGNED_INT, sizeof(unsigned int));
        UNIT_ASSERT_EQUAL(SIZEOF_UNSIGNED_LONG, sizeof(unsigned long));
        UNIT_ASSERT_EQUAL(SIZEOF_UNSIGNED_LONG_LONG, sizeof(unsigned long long));
        UNIT_ASSERT_EQUAL(SIZEOF_UNSIGNED_SHORT, sizeof(unsigned short));
    }
};

UNIT_TEST_SUITE_REGISTRATION(TPlatformTest);
