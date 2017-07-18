#include "dirut.h"
#include "filelist.h"
#include "tempdir.h"

#include <library/unittest/registar.h>

#include <util/system/file.h>
#include <util/generic/string.h>

class TFileListTest: public TTestBase {
    UNIT_TEST_SUITE(TFileListTest);
    UNIT_TEST(TestSimple);
    UNIT_TEST(TestPrefix);
    UNIT_TEST_SUITE_END();

public:
    void TestSimple();
    void TestPrefix();
};

void TFileListTest::TestSimple() {
    TTempDir tempDir("nonexistingdir");
    MakeDirIfNotExist(~(tempDir() + LOCSLASH_S "subdir"));
    TFile(~(tempDir() + LOCSLASH_S "subdir" LOCSLASH_S "file"), CreateAlways);

    TFileList fileList;
    fileList.Fill(~tempDir(), "", "", 1000);
    TString fileName(fileList.Next());
    UNIT_ASSERT_EQUAL(fileName, "subdir" LOCSLASH_S "file");
    UNIT_ASSERT_EQUAL(fileList.Next(), nullptr);
}

void TFileListTest::TestPrefix() {
    TTempDir tempDir("nonexistingdir");
    TFile(~(tempDir() + LOCSLASH_S "good_file1"), CreateAlways);
    TFile(~(tempDir() + LOCSLASH_S "good_file2"), CreateAlways);
    TFile(~(tempDir() + LOCSLASH_S "bad_file1"), CreateAlways);
    TFile(~(tempDir() + LOCSLASH_S "bad_file2"), CreateAlways);

    const bool SORT = true;
    TFileList fileList;
    {
        fileList.Fill(~tempDir(), "good_file", SORT);
        UNIT_ASSERT_EQUAL(TString(fileList.Next()), "good_file1");
        UNIT_ASSERT_EQUAL(TString(fileList.Next()), "good_file2");
        UNIT_ASSERT_EQUAL(fileList.Next(), nullptr);
    }
    {
        fileList.Fill(~tempDir(), "bad_file", SORT);
        UNIT_ASSERT_EQUAL(TString(fileList.Next()), "bad_file1");
        UNIT_ASSERT_EQUAL(TString(fileList.Next()), "bad_file2");
        UNIT_ASSERT_EQUAL(fileList.Next(), nullptr);
    }
}

UNIT_TEST_SUITE_REGISTRATION(TFileListTest);
