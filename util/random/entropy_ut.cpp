#include "entropy.h"

#include <library/cpp/testing/common/env.h>
#include <library/cpp/testing/unittest/registar.h>
#include <util/folder/tempdir.h>
#include <util/system/fs.h>

Y_UNIT_TEST_SUITE(TestEntropy) {
    Y_UNIT_TEST(TestSeed) {
        char buf[100];

        for (size_t i = 0; i < sizeof(buf); ++i) {
            Seed().LoadOrFail(buf, i);
        }
    }

    Y_UNIT_TEST(TestReset) {
        UNIT_ASSERT_NO_EXCEPTION(ResetEntropyPool());
    }

#if !defined(_win_)
    Y_UNIT_TEST(TestMissingWorkingDirectory) {
        NFs::SetCurrentWorkingDirectory(TTempDir::NewTempDir(GetWorkPath()).Name());
        UNIT_ASSERT_NO_EXCEPTION(ResetEntropyPool());
    }
#endif

} // Y_UNIT_TEST_SUITE(TestEntropy)
