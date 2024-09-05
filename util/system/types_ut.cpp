#include "types.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestTypes) {
    Y_UNIT_TEST(TestScanf) {
        i32 val32 = 0;
        sscanf("-123", "%" SCNi32, &val32);
        UNIT_ASSERT(val32 == -123);
        sscanf("234", "%" SCNu32, &val32);
        UNIT_ASSERT(val32 == 234);
        sscanf("159", "%" SCNx32, &val32);
        UNIT_ASSERT(val32 == 345);

        i64 val64 = 0;
        sscanf("-123", "%" SCNi64, &val64);
        UNIT_ASSERT(val64 == -123);
        sscanf("234", "%" SCNu64, &val64);
        UNIT_ASSERT(val64 == 234);
        sscanf("159", "%" SCNx64, &val64);
        UNIT_ASSERT(val64 == 345);
    }
} // Y_UNIT_TEST_SUITE(TestTypes)
