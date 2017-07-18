#include <library/unittest/registar.h>

#include <util/system/fs.h>
#include <util/random/random.h>
#include "direct_io.h"

SIMPLE_UNIT_TEST_SUITE(TDirectIoTestSuite) {
    static const char* FileName_("./test.file");

    SIMPLE_UNIT_TEST(TestDirectFile) {
        TDirectIOBufferedFile file(FileName_, RdWr | Direct | Seq | CreateAlways, 1 << 15);
        yvector<ui64> data((1 << 15) + 1);
        yvector<ui64> readed(data.size());
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
