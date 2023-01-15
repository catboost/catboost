#include "info.h"

#include <library/cpp/testing/unittest/registar.h>

class TSysInfoTest: public TTestBase {
    UNIT_TEST_SUITE(TSysInfoTest);
    UNIT_TEST(TestNumberOfCpus)
    UNIT_TEST(TestGetPageSize)
    UNIT_TEST_SUITE_END();

private:
    inline void TestNumberOfCpus() {
        UNIT_ASSERT(NSystemInfo::NumberOfCpus() > 0);
        UNIT_ASSERT_EQUAL(NSystemInfo::NumberOfCpus(), NSystemInfo::CachedNumberOfCpus());
    }

    inline void TestGetPageSize() {
        UNIT_ASSERT(NSystemInfo::GetPageSize() >= 4096);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TSysInfoTest);
