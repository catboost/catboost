#include "tempfile.h"

#include <library/unittest/registar.h>

#include <util/folder/dirut.h>
#include <util/stream/file.h>

SIMPLE_UNIT_TEST_SUITE(TTempFileHandle) {
    SIMPLE_UNIT_TEST(Create) {
        TString path;
        {
            TTempFileHandle tmp;
            path = tmp.Name();
            tmp.Write("hello world\n", 12);
            tmp.FlushData();
            UNIT_ASSERT_STRINGS_EQUAL(TFileInput(tmp.Name()).ReadAll(), "hello world\n");
        }
        UNIT_ASSERT(!NFs::Exists(path));
    }
}
