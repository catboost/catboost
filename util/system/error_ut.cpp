#include "error.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/ylimits.h>

#ifdef _win_
    #include "winint.h"
#else
    #include <fcntl.h>
#endif

class TSysErrorTest: public TTestBase {
    UNIT_TEST_SUITE(TSysErrorTest);
    UNIT_TEST(TestErrorCode)
    UNIT_TEST(TestErrorMessage)
    UNIT_TEST_SUITE_END();

private:
    inline void TestErrorCode() {
        GenFailure();

        UNIT_ASSERT(LastSystemError() != 0);
    }

    inline void TestErrorMessage() {
        GenFailure();

        UNIT_ASSERT(*LastSystemErrorText() != 0);
    }

    inline void GenFailure() {
#ifdef _win_
        SetLastError(3);
#else
        UNIT_ASSERT(open("/non-existent", O_RDONLY) < 0);
#endif
    }
};

UNIT_TEST_SUITE_REGISTRATION(TSysErrorTest);
