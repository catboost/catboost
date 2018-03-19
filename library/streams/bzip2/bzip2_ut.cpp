#include "bzip2.h"

#include <library/unittest/registar.h>

#include <util/stream/file.h>
#include <util/system/tempfile.h>

#define ZDATA "./zdata"

SIMPLE_UNIT_TEST_SUITE(TBZipTest) {
    static const TString data = "8s7d5vc6s5vc67sa4c65ascx6asd4xcv76adsfxv76s";

    SIMPLE_UNIT_TEST(TestCompress) {
        TUnbufferedFileOutput o(ZDATA);
        TBZipCompress c(&o);

        c.Write(~data, +data);
        c.Finish();
        o.Finish();
    }

    SIMPLE_UNIT_TEST(TestDecompress) {
        TTempFile tmp(ZDATA);

        {
            TUnbufferedFileInput i(ZDATA);
            TBZipDecompress d(&i);

            UNIT_ASSERT_EQUAL(d.ReadLine(), data);
        }
    }

    SIMPLE_UNIT_TEST(TestCorrupted) {
        TMemoryInput i("blablabla", 10);
        TBZipDecompress d(&i);

        UNIT_ASSERT_EXCEPTION(d.ReadLine(), TBZipDecompressError);
    }
}

#undef ZDATA
