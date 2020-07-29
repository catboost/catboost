#include "rotating_file.h"
#include "record.h"

#include <util/generic/string.h>
#include <util/system/fstat.h>
#include <util/system/fs.h>

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/tests_data.h>

Y_UNIT_TEST_SUITE(NewRotatingFileSuite) {
    const TString PATH = GetWorkPath() + "/my.log";

    Y_UNIT_TEST(TestFileWrite) {
        TRotatingFileLogBackend log(PATH, 4000, 2);
        TString data = "my data";
        log.WriteData(TLogRecord(ELogPriority::TLOG_INFO, data.data(), data.size()));
        UNIT_ASSERT_C(TFileStat(PATH).Size > 0, "file " << PATH << " has zero size");
    }

    Y_UNIT_TEST(TestFileRotate) {
        const ui64 maxSize = 40;
        TRotatingFileLogBackend log(PATH, maxSize, 2);
        TStringBuilder data;
        for (size_t i = 0; i < 10; ++i)
            data << "data\n";
        log.WriteData(TLogRecord(ELogPriority::TLOG_INFO, data.data(), data.size()));
        UNIT_ASSERT_C(TFileStat(PATH).Size > 0, "file " << PATH << " has zero size");
        data.clear();
        data << "more data";
        log.WriteData(TLogRecord(ELogPriority::TLOG_INFO, data.data(), data.size()));
        UNIT_ASSERT_C(TFileStat(PATH).Size > 0, "file " << PATH << " has zero size");
        UNIT_ASSERT_C(TFileStat(TStringBuilder{} << PATH << ".1").Size > 0, "file " << PATH << ".1 has zero size");
        UNIT_ASSERT_C(TFileStat(PATH).Size < maxSize, "size of file " << PATH << " is greater than the size limit of " << maxSize << " bytes");
    }

    Y_UNIT_TEST(TestDoubleFileRotate) {
        const ui64 maxSize = 40;
        TRotatingFileLogBackend log(PATH, maxSize, 2);
        TStringBuilder data;
        for (size_t i = 0; i < 10; ++i)
            data << "data\n";
        log.WriteData(TLogRecord(ELogPriority::TLOG_INFO, data.data(), data.size()));
        UNIT_ASSERT_C(TFileStat(PATH).Size > 0, "file " << PATH << " has zero size");
        log.WriteData(TLogRecord(ELogPriority::TLOG_INFO, data.data(), data.size()));
        UNIT_ASSERT_C(TFileStat(PATH).Size > 0, "file " << PATH << " has zero size");
        UNIT_ASSERT_C(TFileStat(TStringBuilder{} << PATH << ".1").Size > 0, "file " << PATH << ".1 has zero size");
        UNIT_ASSERT_C(TFileStat(PATH).Size > maxSize, "size of file " << PATH << " is lesser than was written");
        data.clear();
        data << "more data";
        log.WriteData(TLogRecord(ELogPriority::TLOG_INFO, data.data(), data.size()));
        UNIT_ASSERT_C(TFileStat(PATH).Size > 0, "file " << PATH << " has zero size");
        UNIT_ASSERT_C(TFileStat(TStringBuilder{} << PATH << ".1").Size > 0, "file " << PATH << ".1 has zero size");
        UNIT_ASSERT_C(TFileStat(TStringBuilder{} << PATH << ".2").Size > 0, "file " << PATH << ".2 has zero size");
        UNIT_ASSERT_C(TFileStat(PATH).Size < maxSize, "size of file " << PATH << " is greater than the size limit of " << maxSize << " bytes");
    }
}
