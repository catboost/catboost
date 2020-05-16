#include "execpath.h"

#include <library/cpp/unittest/registar.h>

#include "platform.h"
#include <util/folder/dirut.h>

Y_UNIT_TEST_SUITE(TExecPathTest) {
    Y_UNIT_TEST(TestIt) {
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
