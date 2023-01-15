#include "mem_info.h"

#include <library/cpp/testing/unittest/registar.h>

#include "info.h"

class TMemInfoTest: public NUnitTest::TTestBase {
    UNIT_TEST_SUITE(TMemInfoTest)
    UNIT_TEST(TestMemInfo)
    UNIT_TEST_SUITE_END();
    void TestMemInfo() {
        using namespace NMemInfo;

        TMemInfo stats = GetMemInfo();

        UNIT_ASSERT(stats.RSS >= NSystemInfo::GetPageSize());
        UNIT_ASSERT(stats.VMS >= stats.RSS);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TMemInfoTest)
