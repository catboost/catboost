#include <library/cpp/unittest/registar.h>
#include <library/cpp/archive/yarchive.h>
#include <util/memory/blob.h>

extern "C" {
    extern const ui8 ArchiveAsm[];
    extern const ui32 ArchiveAsmSize;
}

static const unsigned char SimpleArchive[] = {
    #include <tools/archiver/alignment_test/simple_archive.inc>
};


Y_UNIT_TEST_SUITE(AlignmentTest) {
    Y_UNIT_TEST(SimpleArchiveCheck) {
        UNIT_ASSERT_VALUES_EQUAL(size_t(SimpleArchive) % ArchiveWriterDefaultDataAlignment, 0);
        TArchiveReader dataArchive(
            TBlob::NoCopy(SimpleArchive, sizeof(SimpleArchive))
        );
        auto dataFile1 = dataArchive.BlobByKey("/data_file.txt");
        auto dataFile2 = dataArchive.BlobByKey("/data_file2.txt");
        UNIT_ASSERT_NO_DIFF(TStringBuf(dataFile1.AsCharPtr(), dataFile1.Size()), "some text\n");
        UNIT_ASSERT_NO_DIFF(TStringBuf(dataFile2.AsCharPtr(), dataFile2.Size()), "second file content\n");
        UNIT_ASSERT_VALUES_EQUAL(size_t(dataFile1.AsCharPtr()) % ArchiveWriterDefaultDataAlignment, 0);
        UNIT_ASSERT_VALUES_EQUAL(size_t(dataFile2.AsCharPtr()) % ArchiveWriterDefaultDataAlignment, 0);
    }

    Y_UNIT_TEST(ArchiveAsmCheck) {
        UNIT_ASSERT_VALUES_EQUAL(size_t(ArchiveAsm) % ArchiveWriterDefaultDataAlignment, 0);
        TArchiveReader dataArchive(
            TBlob::NoCopy(ArchiveAsm, ArchiveAsmSize)
        );
        auto dataFile1 = dataArchive.BlobByKey("/data_file.txt");
        auto dataFile2 = dataArchive.BlobByKey("/data_file2.txt");
        UNIT_ASSERT_NO_DIFF(TStringBuf(dataFile1.AsCharPtr(), dataFile1.Size()), "some text\n");
        UNIT_ASSERT_NO_DIFF(TStringBuf(dataFile2.AsCharPtr(), dataFile2.Size()), "second file content\n");
        UNIT_ASSERT_VALUES_EQUAL(size_t(dataFile1.AsCharPtr()) % ArchiveWriterDefaultDataAlignment, 0);
        UNIT_ASSERT_VALUES_EQUAL(size_t(dataFile2.AsCharPtr()) % ArchiveWriterDefaultDataAlignment, 0);
    }
}
