#include "file.h"
#include "fs.h"
#include "tempfile.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/file.h>
#include <util/generic/yexception.h>

class TFileTest: public TTestBase {
    UNIT_TEST_SUITE(TFileTest);
    UNIT_TEST(TestOpen);
    UNIT_TEST(TestOpenSync);
    UNIT_TEST(TestRW);
    UNIT_TEST(TestReWrite);
    UNIT_TEST(TestAppend);
    UNIT_TEST(TestLinkTo);
    UNIT_TEST(TestResize);
    UNIT_TEST(TestLocale);
    UNIT_TEST(TestFlush);
    UNIT_TEST(TestFlushSpecialFile);
    UNIT_TEST(TestRawRead);
    UNIT_TEST(TestRead);
    UNIT_TEST(TestRawPread);
    UNIT_TEST(TestPread);
    UNIT_TEST(TestCache);
    UNIT_TEST_SUITE_END();

public:
    void TestOpen();
    void TestOpenSync();
    void TestRW();
    void TestLocale();
    void TestFlush();
    void TestFlushSpecialFile();
    void TestRawRead();
    void TestRead();
    void TestRawPread();
    void TestPread();
    void TestCache();

    inline void TestLinkTo() {
        TTempFile tmp1("tmp1");
        TTempFile tmp2("tmp2");

        {
            TFile f1(tmp1.Name(), OpenAlways | WrOnly);
            TFile f2(tmp2.Name(), OpenAlways | WrOnly);

            f1.LinkTo(f2);

            f1.Write("12345", 5);
            f2.Write("67890", 5);
        }

        UNIT_ASSERT_EQUAL(TUnbufferedFileInput(tmp2.Name()).ReadAll(), "1234567890");
    }

    inline void TestAppend() {
        TTempFile tmp("tmp");

        {
            TFile f(tmp.Name(), OpenAlways | WrOnly);

            f.Write("12345678", 8);
        }

        {
            TFile f(tmp.Name(), OpenAlways | WrOnly | ForAppend);

            f.Write("67", 2);
            f.Write("89", 2);
        }

        UNIT_ASSERT_EQUAL(TUnbufferedFileInput(tmp.Name()).ReadAll(), "123456786789");
    }

    inline void TestReWrite() {
        TTempFile tmp("tmp");

        {
            TFile f(tmp.Name(), OpenAlways | WrOnly);

            f.Write("12345678", 8);
        }

        {
            TFile f(tmp.Name(), OpenAlways | WrOnly);

            f.Write("6789", 4);
        }

        UNIT_ASSERT_EQUAL(TUnbufferedFileInput(tmp.Name()).ReadAll(), "67895678");
    }

    inline void TestResize() {
        TTempFile tmp("tmp");

        {
            TFile file(tmp.Name(), OpenAlways | WrOnly);

            file.Write("1234567", 7);
            file.Seek(3, sSet);

            file.Resize(5);
            UNIT_ASSERT_EQUAL(file.GetLength(), 5);
            UNIT_ASSERT_EQUAL(file.GetPosition(), 3);

            file.Resize(12);
            UNIT_ASSERT_EQUAL(file.GetLength(), 12);
            UNIT_ASSERT_EQUAL(file.GetPosition(), 3);
        }

        const TString data = TUnbufferedFileInput(tmp.Name()).ReadAll();
        UNIT_ASSERT_EQUAL(data.length(), 12);
        UNIT_ASSERT(data.StartsWith("12345"));
    }
};

UNIT_TEST_SUITE_REGISTRATION(TFileTest);

void TFileTest::TestOpen() {
    TString res;
    TFile f1;

    try {
        TFile f2("f1.txt", OpenExisting);
    } catch (const yexception& e) {
        res = e.what();
    }
    UNIT_ASSERT(!res.empty());
    res.remove();

    try {
        TFile f2("f1.txt", OpenAlways);
        f1 = f2;
    } catch (const yexception& e) {
        res = e.what();
    }
    UNIT_ASSERT(res.empty());
    UNIT_ASSERT(f1.IsOpen());
    UNIT_ASSERT_VALUES_EQUAL(f1.GetName(), "f1.txt");
    UNIT_ASSERT_VALUES_EQUAL(f1.GetLength(), 0);

    try {
        TFile f2("f1.txt", CreateNew);
    } catch (const yexception& e) {
        res = e.what();
    }
    UNIT_ASSERT(!res.empty());
    res.remove();

    f1.Close();
    UNIT_ASSERT(unlink("f1.txt") == 0);
}

void TFileTest::TestOpenSync() {
    TFile f1("f1.txt", CreateNew | Sync);
    UNIT_ASSERT(f1.IsOpen());
    f1.Close();
    UNIT_ASSERT(!f1.IsOpen());
    UNIT_ASSERT(unlink("f1.txt") == 0);
}

void TFileTest::TestRW() {
    TFile f1("f1.txt", CreateNew);
    UNIT_ASSERT(f1.IsOpen());
    UNIT_ASSERT_VALUES_EQUAL(f1.GetName(), "f1.txt");
    UNIT_ASSERT_VALUES_EQUAL(f1.GetLength(), 0);
    ui32 d[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    f1.Write(&d, sizeof(ui32) * 10);
    UNIT_ASSERT_VALUES_EQUAL(f1.GetLength(), 40);
    UNIT_ASSERT_VALUES_EQUAL(f1.GetPosition(), 40);
    UNIT_ASSERT_VALUES_EQUAL(f1.Seek(12, sSet), 12);
    f1.Flush();
    ui32 v;
    f1.Load(&v, sizeof(v));
    UNIT_ASSERT_VALUES_EQUAL(v, 3u);
    UNIT_ASSERT_VALUES_EQUAL(f1.GetPosition(), 16);

    TFile f2 = f1;
    UNIT_ASSERT(f2.IsOpen());
    UNIT_ASSERT_VALUES_EQUAL(f2.GetName(), "f1.txt");
    UNIT_ASSERT_VALUES_EQUAL(f2.GetPosition(), 16);
    UNIT_ASSERT_VALUES_EQUAL(f2.GetLength(), 40);
    f2.Write(&v, sizeof(v));

    UNIT_ASSERT_VALUES_EQUAL(f1.GetPosition(), 20);
    UNIT_ASSERT_VALUES_EQUAL(f1.Seek(-4, sCur), 16);
    v = 0;
    f1.Load(&v, sizeof(v));
    UNIT_ASSERT_VALUES_EQUAL(v, 3u);
    f1.Close();
    UNIT_ASSERT(!f1.IsOpen());
    UNIT_ASSERT(!f2.IsOpen());
    UNIT_ASSERT(unlink("f1.txt") == 0);
}

#ifdef _unix_
    #include <locale.h>
#endif

void TFileTest::TestLocale() {
#ifdef _unix_
    const char* loc = setlocale(LC_CTYPE, nullptr);
    setlocale(LC_CTYPE, "ru_RU.UTF-8");
#endif
    TFile f("Имя.txt", CreateNew);
    UNIT_ASSERT(f.IsOpen());
    UNIT_ASSERT_VALUES_EQUAL(f.GetName(), "Имя.txt");
    UNIT_ASSERT_VALUES_EQUAL(f.GetLength(), 0);
    f.Close();
    UNIT_ASSERT(NFs::Remove("Имя.txt"));
#ifdef _unix_
    setlocale(LC_CTYPE, loc);
#endif
}

void TFileTest::TestFlush() {
    TTempFile tmp("tmp");

    {
        TFile f(tmp.Name(), OpenAlways | WrOnly);
        f.Flush();
        f.FlushData();
        f.Close();

        UNIT_ASSERT_EXCEPTION(f.Flush(), TFileError);
        UNIT_ASSERT_EXCEPTION(f.FlushData(), TFileError);
    }
}

void TFileTest::TestFlushSpecialFile() {
#ifdef _unix_
    TFile devNull("/dev/null", WrOnly);
    devNull.FlushData();
    devNull.Flush();
    devNull.Close();
#endif
}

void TFileTest::TestRawRead() {
    TTempFile tmp("tmp");

    {
        TFile file(tmp.Name(), OpenAlways | WrOnly);
        file.Write("1234567", 7);
        file.Flush();
        file.Close();
    }

    {
        TFile file(tmp.Name(), OpenExisting | RdOnly);
        char buf[7];
        i32 reallyRead = file.RawRead(buf, 7);
        Y_ENSURE(0 <= reallyRead && reallyRead <= 7);
        Y_ENSURE(TStringBuf(buf, reallyRead) == TStringBuf("1234567").Head(reallyRead));
    }
}

void TFileTest::TestRead() {
    TTempFile tmp("tmp");

    {
        TFile file(tmp.Name(), OpenAlways | WrOnly);
        file.Write("1234567", 7);
        file.Flush();
        file.Close();
    }

    {
        TFile file(tmp.Name(), OpenExisting | RdOnly);
        char buf[7];
        Y_ENSURE(file.Read(buf, 7) == 7);
        Y_ENSURE(TStringBuf(buf, 7) == "1234567");

        memset(buf, 0, sizeof(buf));
        file.Seek(0, sSet);
        Y_ENSURE(file.Read(buf, 123) == 7);
        Y_ENSURE(TStringBuf(buf, 7) == "1234567");
    }
}

void TFileTest::TestRawPread() {
    TTempFile tmp("tmp");

    {
        TFile file(tmp.Name(), OpenAlways | WrOnly);
        file.Write("1234567", 7);
        file.Flush();
        file.Close();
    }

    {
        TFile file(tmp.Name(), OpenExisting | RdOnly);
        char buf[7];
        i32 reallyRead = file.RawPread(buf, 3, 1);
        Y_ENSURE(0 <= reallyRead && reallyRead <= 3);
        Y_ENSURE(TStringBuf(buf, reallyRead) == TStringBuf("234").Head(reallyRead));

        memset(buf, 0, sizeof(buf));
        reallyRead = file.RawPread(buf, 2, 5);
        Y_ENSURE(0 <= reallyRead && reallyRead <= 2);
        Y_ENSURE(TStringBuf(buf, reallyRead) == TStringBuf("67").Head(reallyRead));
    }
}

void TFileTest::TestPread() {
    TTempFile tmp("tmp");

    {
        TFile file(tmp.Name(), OpenAlways | WrOnly);
        file.Write("1234567", 7);
        file.Flush();
        file.Close();
    }

    {
        TFile file(tmp.Name(), OpenExisting | RdOnly);
        char buf[7];
        Y_ENSURE(file.Pread(buf, 3, 1) == 3);
        Y_ENSURE(TStringBuf(buf, 3) == "234");

        memset(buf, 0, sizeof(buf));
        Y_ENSURE(file.Pread(buf, 2, 5) == 2);
        Y_ENSURE(TStringBuf(buf, 2) == "67");
    }
}

#ifdef _linux_
    #include <sys/statfs.h>
#endif

#ifndef TMPFS_MAGIC
    #define TMPFS_MAGIC 0x01021994
#endif

void TFileTest::TestCache() {
#ifdef _linux_
    { // create file in /tmp, current dir could be tmpfs which does not support fadvise
        TFile file(MakeTempName("/tmp"), OpenAlways | Transient | RdWr | NoReadAhead);

        struct statfs fs;
        if (!fstatfs(file.GetHandle(), &fs) && fs.f_type == TMPFS_MAGIC) {
            return;
        }

        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(), 0);
        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(0, 0), 0);

        file.Resize(7);
        file.PrefetchCache();
        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(), 7);
        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(3, 2), 2);

        file.FlushCache();
        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(), 7);

        file.EvictCache();
        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(), 0);

        file.PrefetchCache();
        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(), 7);

        file.Resize(12345);
        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(), 4096);
        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(4096, 0), 0);

        file.PrefetchCache();
        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(), 12345);

        file.FlushCache();
        file.EvictCache();
        UNIT_ASSERT_LE(file.CountCache(), 0);

        file.Resize(33333333);
        file.PrefetchCache(11111111, 11111111);
        UNIT_ASSERT_GE(file.CountCache(), 11111111);

        UNIT_ASSERT_LE(file.CountCache(0, 11111111), 1111111);
        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(11111111, 11111111), 11111111);
        UNIT_ASSERT_LE(file.CountCache(22222222, 11111111), 1111111);

        file.FlushCache(11111111, 11111111);
        UNIT_ASSERT_GE(file.CountCache(), 11111111);

        // first and last incomplete pages could stay in cache
        file.EvictCache(11111111, 11111111);
        UNIT_ASSERT_LT(file.CountCache(11111111, 11111111), 4096 * 2);

        file.EvictCache();
        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(), 0);
    }
#else
    {
        TFile file(MakeTempName(), OpenAlways | Transient | RdWr);

        file.Resize(12345);

        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(), -1);
        file.PrefetchCache();
        file.FlushCache();
        file.EvictCache();
        UNIT_ASSERT_VALUES_EQUAL(file.CountCache(0, 12345), -1);
    }
#endif
}

Y_UNIT_TEST_SUITE(TTestFileHandle) {
    Y_UNIT_TEST(MoveAssignment) {
        TTempFile tmp("tmp");
        {
            TFileHandle file1(tmp.Name(), OpenAlways | WrOnly);
            file1.Write("1", 1);

            TFileHandle file2;
            file2 = std::move(file1);
            Y_ENSURE(!file1.IsOpen());
            Y_ENSURE(file2.IsOpen());

            file2.Write("2", 1);
        }

        {
            TFileHandle file(tmp.Name(), OpenExisting | RdOnly);
            char buf[2];
            Y_ENSURE(file.Read(buf, 2) == 2);
            Y_ENSURE(TStringBuf(buf, 2) == "12");
        }
    }
} // Y_UNIT_TEST_SUITE(TTestFileHandle)

Y_UNIT_TEST_SUITE(TTestDecodeOpenMode) {
    Y_UNIT_TEST(It) {
        UNIT_ASSERT_VALUES_EQUAL("0", DecodeOpenMode(0));
        UNIT_ASSERT_VALUES_EQUAL("RdOnly", DecodeOpenMode(RdOnly));
        UNIT_ASSERT_VALUES_EQUAL("RdWr", DecodeOpenMode(RdWr));
        UNIT_ASSERT_VALUES_EQUAL("WrOnly|ForAppend", DecodeOpenMode(WrOnly | ForAppend));
        UNIT_ASSERT_VALUES_EQUAL("RdWr|CreateAlways|CreateNew|ForAppend|Transient|CloseOnExec|Temp|Sync|Direct|DirectAligned|Seq|NoReuse|NoReadAhead|AX|AR|AW|AWOther|0xF8888000", DecodeOpenMode(0xFFFFFFFF));
    }
} // Y_UNIT_TEST_SUITE(TTestDecodeOpenMode)
