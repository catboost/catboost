#include "all.h"

#include <library/cpp/unittest/registar.h>

#include <util/system/fs.h>
#include <util/system/rwlock.h>
#include <util/system/yield.h>
#include <util/memory/blob.h>
#include <util/stream/file.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

class TLogTest: public TTestBase {
    UNIT_TEST_SUITE(TLogTest);
    UNIT_TEST(TestFile)
    UNIT_TEST(TestFormat)
    UNIT_TEST(TestWrite)
    UNIT_TEST(TestThreaded)
    UNIT_TEST(TestThreadedWithOverflow)
    UNIT_TEST(TestNoFlush)
    UNIT_TEST_SUITE_END();

private:
    void TestFile();
    void TestFormat();
    void TestWrite();
    void TestThreaded();
    void TestThreadedWithOverflow();
    void TestNoFlush();
    void SetUp() override;
    void TearDown() override;
};

UNIT_TEST_SUITE_REGISTRATION(TLogTest);

#define LOGFILE "tmplogfile"

void TLogTest::TestFile() {
    {
        TLog log;

        {
            TLog filelog(LOGFILE);

            log = filelog;
        }

        int v1 = 12;
        unsigned v2 = 34;
        double v3 = 3.0;
        const char* v4 = "qwqwqw";

        log.ReopenLog();
        log.AddLog("some useful data %d, %u, %lf, %s\n", v1, v2, v3, v4);
    }

    TBlob data = TBlob::FromFileSingleThreaded(LOGFILE);

    UNIT_ASSERT_EQUAL(TString((const char*)data.Begin(), data.Size()), "some useful data 12, 34, 3.000000, qwqwqw\n");
}

void TLogTest::TestThreaded() {
    {
        TFileLogBackend fb(LOGFILE);
        TLog log(new TThreadedLogBackend(&fb));

        int v1 = 12;
        unsigned v2 = 34;
        double v3 = 3.0;
        const char* v4 = "qwqwqw";

        log.ReopenLog();
        log.AddLog("some useful data %d, %u, %lf, %s\n", v1, v2, v3, v4);
    }

    TBlob data = TBlob::FromFileSingleThreaded(LOGFILE);

    UNIT_ASSERT_EQUAL(TString((const char*)data.Begin(), data.Size()), "some useful data 12, 34, 3.000000, qwqwqw\n");
}

void TLogTest::TestThreadedWithOverflow() {
    class TFakeLogBackend: public TLogBackend {
    public:
        TWriteGuard Guard() {
            return TWriteGuard(Lock_);
        }

        void WriteData(const TLogRecord&) override {
            TReadGuard guard(Lock_);
        }

        void ReopenLog() override {
            TWriteGuard guard(Lock_);
        }

    private:
        TRWMutex Lock_;
    };

    auto waitForFreeQueue = [](const TLog& log) {
        ThreadYield();
        while (log.BackEndQueueSize() > 0) {
            Sleep(TDuration::MilliSeconds(1));
        }
    };

    TFakeLogBackend fb;
    {
        TLog log(new TThreadedLogBackend(&fb, 2));

        auto guard = fb.Guard();
        log.AddLog("first write");
        waitForFreeQueue(log);
        log.AddLog("second write (first in queue)");
        log.AddLog("third write (second in queue)");
        UNIT_ASSERT_EXCEPTION(log.AddLog("fourth write (queue overflow)"), yexception);
    }

    {
        ui32 overflows = 0;
        TLog log(new TThreadedLogBackend(&fb, 2, [&overflows] { ++overflows; }));

        auto guard = fb.Guard();
        log.AddLog("first write");
        waitForFreeQueue(log);
        log.AddLog("second write (first in queue)");
        log.AddLog("third write (second in queue)");
        UNIT_ASSERT_EQUAL(overflows, 0);
        log.AddLog("fourth write (queue overflow)");
        UNIT_ASSERT_EQUAL(overflows, 1);
    }
}

void TLogTest::TestNoFlush() {
    {
        TFileLogBackend fb(LOGFILE);
        TLog log(new TThreadedLogBackend(&fb));

        int v1 = 12;
        unsigned v2 = 34;
        double v3 = 3.0;
        const char* v4 = "qwqwqw";

        log.ReopenLogNoFlush();
        log.AddLog("some useful data %d, %u, %lf, %s\n", v1, v2, v3, v4);
    }

    TBlob data = TBlob::FromFileSingleThreaded(LOGFILE);

    UNIT_ASSERT_EQUAL(TString((const char*)data.Begin(), data.Size()), "some useful data 12, 34, 3.000000, qwqwqw\n");
}

void TLogTest::TestFormat() {
    TStringStream data;

    {
        TLog log(new TStreamLogBackend(&data));

        log << "qw"
            << " "
            << "1234" << 1234 << " " << 12.3 << 'q' << Endl;
    }

    UNIT_ASSERT_EQUAL(data.Str(), "qw 12341234 12.3q\n");
}

void TLogTest::TestWrite() {
    TStringStream data;
    TString test;

    {
        TLog log(new TStreamLogBackend(&data));

        for (size_t i = 0; i < 1000; ++i) {
            TVector<char> buf(i, (char)i);

            test.append(buf.data(), buf.size());
            log.Write(buf.data(), buf.size());
        }
    }

    UNIT_ASSERT_EQUAL(data.Str(), test);
}

void TLogTest::SetUp() {
    TearDown();
}

void TLogTest::TearDown() {
    NFs::Remove(LOGFILE);
}
