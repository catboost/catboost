#include "directory_models_archive_reader.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/folder/tempdir.h>
#include <util/string/cast.h>
#include <util/stream/file.h>
#include <util/system/tempfile.h>
#include <util/memory/blob.h>

class TDirectoryModelsArchiveReaderTest: public TTestBase {
    UNIT_TEST_SUITE(TDirectoryModelsArchiveReaderTest)
    UNIT_TEST(TestRead);
    UNIT_TEST_SUITE_END();

private:
    void TestRead();
};

UNIT_TEST_SUITE_REGISTRATION(TDirectoryModelsArchiveReaderTest);

const TString MAIN_DIR = "./dir";
const TString SUBDIR = "/subdir";
const TString SAMPLE_FILE1 = "/sample1";
const TString SAMPLE_FILE2 = "/sample2";
const TString TEST_TEXT = "Test Text.";

void TDirectoryModelsArchiveReaderTest::TestRead() {
    TTempDir mainDir(MAIN_DIR);
    TTempDir subDir(MAIN_DIR + SUBDIR);
    TTempFileHandle file1(MAIN_DIR + SAMPLE_FILE1);
    TTempFileHandle file2(MAIN_DIR + SUBDIR + SAMPLE_FILE2);

    file1.Write(TEST_TEXT.data(), TEST_TEXT.size());
    file1.FlushData();

    TDirectoryModelsArchiveReader reader(MAIN_DIR, false);

    UNIT_ASSERT_EQUAL(reader.Count(), 2);

    UNIT_ASSERT(reader.Has(SAMPLE_FILE1));
    UNIT_ASSERT(reader.Has(SUBDIR + SAMPLE_FILE2));

    UNIT_ASSERT_EQUAL(reader.KeyByIndex(0), SAMPLE_FILE1);
    UNIT_ASSERT_EQUAL(reader.KeyByIndex(1), SUBDIR + SAMPLE_FILE2);

    TBlob blob = reader.BlobByKey(SAMPLE_FILE1);
    Cout << "'" << TString(blob.AsCharPtr(), blob.Size()) << "' - '" << TEST_TEXT << "'" << Endl;
    UNIT_ASSERT_VALUES_EQUAL(TString(blob.AsCharPtr(), blob.Size()), TString(TEST_TEXT));

    TAutoPtr<IInputStream> is = reader.ObjectByKey(SAMPLE_FILE1);
    const TString data = is->ReadAll();
    Cout << "'" << data << "' - '" << TEST_TEXT << "'" << Endl;
    UNIT_ASSERT_VALUES_EQUAL(data, TEST_TEXT);
}
