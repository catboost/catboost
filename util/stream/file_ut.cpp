#include "file.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/system/tempfile.h>

static const char* TmpFileName = "./fileio";
static const char* TmpFileContents = "To do good to Mankind is the chivalrous plan";
static const char* TmpFileSubstring = strstr(TmpFileContents, "chivalrous");

Y_UNIT_TEST_SUITE(TFileTest) {
    Y_UNIT_TEST(InputTest) {
        TTempFile tmp(TmpFileName);

        {
            TUnbufferedFileOutput output(TmpFileName);
            output.Write(TmpFileContents, strlen(TmpFileContents));
        }

        {
            TUnbufferedFileInput input(TmpFileName);
            TString s = input.ReadAll();
            UNIT_ASSERT_VALUES_EQUAL(s, TmpFileContents);
        }

        {
            TUnbufferedFileInput input(TmpFileName);
            input.Skip(TmpFileSubstring - TmpFileContents);
            TString s = input.ReadAll();
            UNIT_ASSERT_VALUES_EQUAL(s, "chivalrous plan");
        }

        {
            TUnbufferedFileOutput output(TFile::ForAppend(TmpFileName));
            output.Write(TmpFileContents, strlen(TmpFileContents));
        }

        {
            TUnbufferedFileInput input(TmpFileName);
            TString s = input.ReadAll();
            UNIT_ASSERT_VALUES_EQUAL(s, TString::Join(TmpFileContents, TmpFileContents));
        }
    }

    Y_UNIT_TEST(EmptyMapTest) {
        TTempFile tmp(TmpFileName);

        {
            TUnbufferedFileOutput output(TmpFileName);
            /* Write nothing. */
        }

        {
            TMappedFileInput input(TmpFileName);
            TString s = input.ReadAll();
            UNIT_ASSERT(s.empty());
        }
    }

#ifdef _unix_
    Y_UNIT_TEST(PipeReadLineTest) {
        int fds[2];
        UNIT_ASSERT(pipe(fds) == 0);
        TFile readEnd(fds[0]);
        TFileInput fileInput(readEnd);
        UNIT_ASSERT_VALUES_EQUAL(write(fds[1], "hello\n", 6), 6);

        TString line;
        UNIT_ASSERT(fileInput.ReadLine(line));
        UNIT_ASSERT_STRINGS_EQUAL(line, "hello");
        close(fds[1]);
    }
#endif
} // Y_UNIT_TEST_SUITE(TFileTest)
