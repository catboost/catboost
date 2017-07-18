#include "file.h"
#include "fs.h"

#include <library/unittest/registar.h>

#include <util/stream/file.h>
#include "tempfile.h"
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
    UNIT_TEST_SUITE_END();

public:
    void TestOpen();
    void TestOpenSync();
    void TestRW();
    void TestLocale();
    void TestFlush();
    void TestFlushSpecialFile();

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

        UNIT_ASSERT_EQUAL(TFileInput(tmp2.Name()).ReadAll(), "1234567890");
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

        UNIT_ASSERT_EQUAL(TFileInput(tmp.Name()).ReadAll(), "123456786789");
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

        UNIT_ASSERT_EQUAL(TFileInput(tmp.Name()).ReadAll(), "67895678");
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

        const TString data = TFileInput(tmp.Name()).ReadAll();
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

SIMPLE_UNIT_TEST_SUITE(TTestDecodeOpenMode) {
    SIMPLE_UNIT_TEST(It) {
        UNIT_ASSERT_VALUES_EQUAL("0", DecodeOpenMode(0));
        UNIT_ASSERT_VALUES_EQUAL("RdOnly", DecodeOpenMode(RdOnly));
        UNIT_ASSERT_VALUES_EQUAL("RdWr", DecodeOpenMode(RdWr));
        UNIT_ASSERT_VALUES_EQUAL("WrOnly|ForAppend", DecodeOpenMode(WrOnly | ForAppend));
        UNIT_ASSERT_VALUES_EQUAL("RdWr|CreateNew|AX|AR|AW|CreateAlways|Seq|Direct|Temp|ForAppend|Transient|DirectAligned|AWOther|0xF888EC00", DecodeOpenMode(0xFFFFFFFF));
    }
}
