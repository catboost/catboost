#include "progname.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TProgramNameTest) {
    SIMPLE_UNIT_TEST(TestIt) {
        TString progName = GetProgramName();

        try {
            UNIT_ASSERT(
                progName.find("ut_util") != TString::npos || progName.find("util-system_ut") != TString::npos || progName.find("util-system-ut") != TString::npos);
        } catch (...) {
            Cerr << progName << Endl;

            throw;
        }
    }
}
