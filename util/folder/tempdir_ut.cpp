#include "tempdir.h"

#include <library/cpp/testing/common/env.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/maybe.h>

Y_UNIT_TEST_SUITE(TTempDirTests) {
    Y_UNIT_TEST(TestMoveCtor) {
        TMaybe<TTempDir> dir{TTempDir::NewTempDir(GetWorkPath())};
        UNIT_ASSERT_NO_EXCEPTION(dir.Clear());
    }
} // Y_UNIT_TEST_SUITE(TTempDirTests)
