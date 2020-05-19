#include <library/cpp/unittest/registar.h>

#include <util/generic/yexception.h>
#include <util/system/fs.h>
#include <util/random/random.h>

#include "direct_io.h"

static const char* FileName_("./test.file");

Y_UNIT_TEST_SUITE(TDirectIoTestSuite) {

    Y_UNIT_TEST(TestDirectFile) {
        TDirectIOBufferedFile file(FileName_, RdWr | Direct | Seq | CreateAlways, 1 << 15);
        TVector<ui64> data((1 << 15) + 1);
        TVector<ui64> readed(data.size());
        for (auto& i : data)
            i = RandomNumber<ui64>();
        for (size_t writePos = 0; writePos < data.size();) {
            size_t writeCount = Min<size_t>(1 + RandomNumber<size_t>(1 << 10), data.ysize() - writePos);
            file.Write(&data[writePos], sizeof(ui64) * writeCount);
            writePos += writeCount;
            size_t readPos = RandomNumber(writePos);
            size_t readCount = RandomNumber(writePos - readPos);
            UNIT_ASSERT_VALUES_EQUAL(
                file.Pread(&readed[0], readCount * sizeof(ui64), readPos * sizeof(ui64)),
                readCount * sizeof(ui64));
            for (size_t i = 0; i < readCount; ++i) {
                UNIT_ASSERT_VALUES_EQUAL(readed[i], data[i + readPos]);
            }
        }
        file.Finish();
        TDirectIOBufferedFile fileNew(FileName_, RdOnly | Direct | Seq | OpenAlways, 1 << 15);
        for (int i = 0; i < 1000; ++i) {
            size_t readPos = RandomNumber(data.size());
            size_t readCount = RandomNumber(data.size() - readPos);
            UNIT_ASSERT_VALUES_EQUAL(
                fileNew.Pread(&readed[0], readCount * sizeof(ui64), readPos * sizeof(ui64)),
                readCount * sizeof(ui64));
            for (size_t j = 0; j < readCount; ++j)
                UNIT_ASSERT_VALUES_EQUAL(readed[j], data[j + readPos]);
        }
        size_t readCount = data.size();
        UNIT_ASSERT_VALUES_EQUAL(
            fileNew.Pread(&readed[0], readCount * sizeof(ui64), 0),
            readCount * sizeof(ui64));
        for (size_t i = 0; i < readCount; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(readed[i], data[i]);
        }
        NFs::Remove(FileName_);
    }

}

Y_UNIT_TEST_SUITE(TDirectIoErrorHandling) {
    Y_UNIT_TEST(Constructor) {
        // A non-existent file should not be opened for reading
        UNIT_ASSERT_EXCEPTION(TDirectIOBufferedFile (FileName_, RdOnly, 1 << 15), TFileError);
    }

    Y_UNIT_TEST(WritingReadOnlyFileBufferFlushed) {
        // Note the absence of Direct
        TDirectIOBufferedFile file(FileName_, RdOnly | OpenAlways, 1);
        TString buffer = "Hello";
        UNIT_ASSERT_EXCEPTION(file.Write(buffer.data(), buffer.size()), TFileError);
        NFs::Remove(FileName_);
    }

    Y_UNIT_TEST(WritingReadOnlyFileAllInBuffer) {
        TDirectIOBufferedFile file(FileName_, RdOnly | Direct | Seq | OpenAlways, 1 << 15);
        TString buffer = "Hello";

        // Doesn't throw because of buffering.
        file.Write(buffer.data(), buffer.size());

        UNIT_ASSERT_EXCEPTION(file.Finish(), TFileError);
        NFs::Remove(FileName_);
    }

}
