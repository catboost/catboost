#include "tempfile.h"

#include <library/cpp/unittest/registar.h>

#include <util/folder/dirut.h>
#include <util/stream/file.h>

Y_UNIT_TEST_SUITE(TTempFileHandle) {
    Y_UNIT_TEST(Create) {
        TString path;
        {
            TTempFileHandle tmp;
            path = tmp.Name();
            tmp.Write("hello world\n", 12);
            tmp.FlushData();
            UNIT_ASSERT_STRINGS_EQUAL(TUnbufferedFileInput(tmp.Name()).ReadAll(), "hello world\n");
        }
        UNIT_ASSERT(!NFs::Exists(path));
    }
}
