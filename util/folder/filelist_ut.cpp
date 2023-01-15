#include "dirut.h"
#include "filelist.h"
#include "tempdir.h"

#include <library/cpp/testing/unittest/registar.h>

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
    MakeDirIfNotExist((tempDir() + LOCSLASH_S "subdir").data());
    TFile((tempDir() + LOCSLASH_S "subdir" LOCSLASH_S "file").data(), CreateAlways);

    TFileList fileList;
    fileList.Fill(tempDir().data(), "", "", 1000);
    TString fileName(fileList.Next());
    UNIT_ASSERT_EQUAL(fileName, "subdir" LOCSLASH_S "file");
    UNIT_ASSERT_EQUAL(fileList.Next(), nullptr);
}

void TFileListTest::TestPrefix() {
    TTempDir tempDir("nonexistingdir");
    TFile((tempDir() + LOCSLASH_S "good_file1").data(), CreateAlways);
    TFile((tempDir() + LOCSLASH_S "good_file2").data(), CreateAlways);
    TFile((tempDir() + LOCSLASH_S "bad_file1").data(), CreateAlways);
    TFile((tempDir() + LOCSLASH_S "bad_file2").data(), CreateAlways);

    const bool SORT = true;
    TFileList fileList;
    {
        fileList.Fill(tempDir().data(), "good_file", SORT);
        UNIT_ASSERT_EQUAL(TString(fileList.Next()), "good_file1");
        UNIT_ASSERT_EQUAL(TString(fileList.Next()), "good_file2");
        UNIT_ASSERT_EQUAL(fileList.Next(), nullptr);
    }
    {
        fileList.Fill(tempDir().data(), "bad_file", SORT);
        UNIT_ASSERT_EQUAL(TString(fileList.Next()), "bad_file1");
        UNIT_ASSERT_EQUAL(TString(fileList.Next()), "bad_file2");
        UNIT_ASSERT_EQUAL(fileList.Next(), nullptr);
    }
}

UNIT_TEST_SUITE_REGISTRATION(TFileListTest);
