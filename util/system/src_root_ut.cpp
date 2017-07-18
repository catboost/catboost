#include "src_root.h"

#include <util/folder/pathsplit.h>
#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TestSourceRoot) {
    SIMPLE_UNIT_TEST(TestStrip) {
        // Reconstruct() converts "\" -> "/" on Windows
        const TString path = TPathSplit(__SOURCE_FILE_IMPL__.As<TStringBuf>()).Reconstruct();
        UNIT_ASSERT_EQUAL(path, "util" LOCSLASH_S "system" LOCSLASH_S "src_root_ut.cpp");
    }
}
