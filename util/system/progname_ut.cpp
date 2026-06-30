#include "progname.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TProgramNameTest) {
    Y_UNIT_TEST(TestIt) {
        TString progName = GetProgramName();

        try {
            UNIT_ASSERT(
                progName.find("ut_util") != TString::npos || progName.find("util-system_ut") != TString::npos || progName.find("util-system-ut") != TString::npos);
        } catch (...) {
            Cerr << progName << Endl;

            throw;
        }
    }
} // Y_UNIT_TEST_SUITE(TProgramNameTest)
