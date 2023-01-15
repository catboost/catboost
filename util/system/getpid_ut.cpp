#include "getpid.h"

#include <library/cpp/testing/unittest/registar.h>

class TGetPidTest: public TTestBase {
    UNIT_TEST_SUITE(TGetPidTest);
    UNIT_TEST(Test);
    UNIT_TEST_SUITE_END();

public:
    void Test();
};

UNIT_TEST_SUITE_REGISTRATION(TGetPidTest);

void TGetPidTest::Test() {
    const TProcessId pid = GetPID();
    UNIT_ASSERT(pid != 0);
}
