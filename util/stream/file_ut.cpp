#include "file.h"

#include <library/unittest/registar.h>

#include <util/system/tempfile.h>

static const char* TmpFileName = "./fileio";
static const char* TmpFileContents = "To do good to Mankind is the chivalrous plan";
static const char* TmpFileSubstring = strstr(TmpFileContents, "chivalrous");

SIMPLE_UNIT_TEST_SUITE(TFileTest) {
    SIMPLE_UNIT_TEST(InputTest) {
        TTempFile tmp(TmpFileName);

        {
            TFileOutput output(TmpFileName);
            output.Write(TmpFileContents, strlen(TmpFileContents));
        }

        {
            TFileInput input(TmpFileName);
            TString s = input.ReadAll();
            UNIT_ASSERT_VALUES_EQUAL(s, TmpFileContents);
        }

        {
            TFileInput input(TmpFileName);
            input.Skip(TmpFileSubstring - TmpFileContents);
            TString s = input.ReadAll();
            UNIT_ASSERT_VALUES_EQUAL(s, "chivalrous plan");
        }

        {
            TFileOutput output(TFile::ForAppend(TmpFileName));
            output.Write(TmpFileContents, strlen(TmpFileContents));
        }

        {
            TFileInput input(TmpFileName);
            TString s = input.ReadAll();
            UNIT_ASSERT_VALUES_EQUAL(s, TString::Join(TmpFileContents, TmpFileContents));
        }
    }

    SIMPLE_UNIT_TEST(EmptyMapTest) {
        TTempFile tmp(TmpFileName);

        {
            TFileOutput output(TmpFileName);
            /* Write nothing. */
        }

        {
            TMappedFileInput input(TmpFileName);
            TString s = input.ReadAll();
            UNIT_ASSERT(s.empty());
        }
    }
}
