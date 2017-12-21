#include "execpath.h"

#include <library/unittest/registar.h>

#include "platform.h"
#include <util/folder/dirut.h>

SIMPLE_UNIT_TEST_SUITE(TExecPathTest) {
    SIMPLE_UNIT_TEST(TestIt) {
        TString execPath = GetExecPath();
        TString persistentExecPath = GetPersistentExecPath();

        try {
            UNIT_ASSERT(NFs::Exists(execPath));
            UNIT_ASSERT(NFs::Exists(persistentExecPath));
        } catch (...) {
            Cerr << execPath << Endl;

            throw;
        }
    }
}
