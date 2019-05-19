#include "rotating_file.h"
#include "record.h"

#include <util/generic/string.h>
#include <util/system/fstat.h>
#include <util/system/fs.h>

#include <library/unittest/registar.h>
#include <library/unittest/tests_data.h>

Y_UNIT_TEST_SUITE(RotatingFileSuite) {
    const TString PRE_ROTATION_PATH = GetWorkPath() + "/new.log";
    const TString POST_ROTATION_PATH = GetWorkPath() + "/old.log";

    Y_UNIT_TEST(TestFileWrite) {
        TRotatingFileLogBackend log(PRE_ROTATION_PATH, POST_ROTATION_PATH, 4000);
        TString data = "my data";
        log.WriteData(TLogRecord(ELogPriority::TLOG_INFO, data.data(), data.size()));
        UNIT_ASSERT_C(TFileStat(PRE_ROTATION_PATH).Size > 0, "file " << PRE_ROTATION_PATH << " has zero size");
    }

    Y_UNIT_TEST(TestFileRotate) {
        const ui64 maxSize = 40;
        TRotatingFileLogBackend log(PRE_ROTATION_PATH, POST_ROTATION_PATH, maxSize);
        TStringBuilder data;
        for (size_t i = 0; i < 10; ++i)
            data << "data\n";
        log.WriteData(TLogRecord(ELogPriority::TLOG_INFO, data.data(), data.size()));
        UNIT_ASSERT_C(TFileStat(PRE_ROTATION_PATH).Size > 0, "file " << PRE_ROTATION_PATH << " has zero size");
        data.clear();
        data << "more data";
        log.WriteData(TLogRecord(ELogPriority::TLOG_INFO, data.data(), data.size()));
        UNIT_ASSERT_C(TFileStat(PRE_ROTATION_PATH).Size > 0, "file " << PRE_ROTATION_PATH << " has zero size");
        UNIT_ASSERT_C(TFileStat(POST_ROTATION_PATH).Size > 0, "file " << POST_ROTATION_PATH << " has zero size");
        UNIT_ASSERT_C(TFileStat(PRE_ROTATION_PATH).Size < maxSize, "size of file " << PRE_ROTATION_PATH << " is greater than the size limit of " << maxSize << " bytes");
    }

    Y_UNIT_TEST(TestBigFileRotate) {
        const ui64 maxSize = 40;
        TRotatingFileLogBackend log(PRE_ROTATION_PATH, POST_ROTATION_PATH, maxSize);
        TStringBuilder data;
        for (size_t i = 0; i < 10; ++i)
            data << "data\n";
        log.WriteData(TLogRecord(ELogPriority::TLOG_INFO, data.data(), data.size()));
        UNIT_ASSERT_C(TFileStat(PRE_ROTATION_PATH).Size > 0, "file " << PRE_ROTATION_PATH << " has zero size");
        log.WriteData(TLogRecord(ELogPriority::TLOG_INFO, data.data(), data.size()));
        UNIT_ASSERT_C(TFileStat(PRE_ROTATION_PATH).Size > 0, "file " << PRE_ROTATION_PATH << " has zero size");
        UNIT_ASSERT_C(TFileStat(POST_ROTATION_PATH).Size > 0, "file " << POST_ROTATION_PATH << " has zero size");
        UNIT_ASSERT_C(TFileStat(PRE_ROTATION_PATH).Size > maxSize, "size of file " << PRE_ROTATION_PATH << " is lesser than was written");
    }
}
