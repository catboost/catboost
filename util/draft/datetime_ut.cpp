#include "datetime.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/string/builder.h>

Y_UNIT_TEST_SUITE(TSimpleTMTest) {
    TString PrintMarker(const TString& test, int line) {
        return TStringBuilder() << "test " << test << " at line " << line;
    }

    TString JoinMarker(const TString& marker, time_t t) {
        return TStringBuilder() << marker << " (tstamp=" << t << ")";
    }

    TString PrintMarker(const TString& test, int line, time_t t) {
        return JoinMarker(PrintMarker(test, line), t);
    }

    void AssertStructTmEqual(const TString& marker, const struct tm& tmt, const NDatetime::TSimpleTM& tms) {
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.Sec, tmt.tm_sec, marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.Min, tmt.tm_min, marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.Hour, tmt.tm_hour, marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.WDay, tmt.tm_wday, marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.MDay, tmt.tm_mday, marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.Mon, tmt.tm_mon, marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.YDay, tmt.tm_yday, marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.Year, tmt.tm_year, marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.IsDst, tmt.tm_isdst, marker);
#ifndef _win_
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.GMTOff, tmt.tm_gmtoff, marker);
#endif
    }

    void AssertSimpleTM(const TString& mark,
                        const NDatetime::TSimpleTM& tms,
                        time_t tstamp, ui32 year, ui32 mon, ui32 mday, ui32 hour, ui32 minu, ui32 sec) {
        TString marker = JoinMarker(mark, tstamp);
        struct tm tmt;
        Zero(tmt);
        GmTimeR(&tstamp, &tmt);
        AssertStructTmEqual(marker, tmt, tms);
        tmt = tms.AsStructTmUTC();
        time_t tstamp1 = TimeGM(&tmt);
        UNIT_ASSERT_VALUES_EQUAL_C(tstamp, tstamp1, marker);
        UNIT_ASSERT_VALUES_EQUAL_C(tstamp, tms.AsTimeT(), marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.RealYear(), year, marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.RealMonth(), mon, marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.MDay, mday, marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.Hour, hour, marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.Min, minu, marker);
        UNIT_ASSERT_VALUES_EQUAL_C((int)tms.Sec, sec, marker);
    }

    Y_UNIT_TEST(TestLeap) {
        using namespace NDatetime;
        UNIT_ASSERT(LeapYearAD(2000));
        UNIT_ASSERT(LeapYearAD(2012));
        UNIT_ASSERT(!LeapYearAD(1999));
        UNIT_ASSERT(LeapYearAD(2004));
        UNIT_ASSERT(!LeapYearAD(1900));
    }

    Y_UNIT_TEST(TestYDayConversion) {
        using namespace NDatetime;
        ui32 month;
        ui32 mday;

        for (ui32 yday = 0; yday < 365; ++yday) {
            YDayToMonthAndDay(yday, false, &month, &mday);
            UNIT_ASSERT_VALUES_EQUAL(yday, YDayFromMonthAndDay(month, mday, false));
        }
        for (ui32 yday = 0; yday < 366; ++yday) {
            YDayToMonthAndDay(yday, true, &month, &mday);
            UNIT_ASSERT_VALUES_EQUAL(yday, YDayFromMonthAndDay(month, mday, true));
        }

        UNIT_ASSERT_EXCEPTION(YDayToMonthAndDay(365, false, &month, &mday), yexception);
        UNIT_ASSERT_EXCEPTION(YDayToMonthAndDay(366, true, &month, &mday), yexception);
    }

    Y_UNIT_TEST(SimpleTMTest) {
        using namespace NDatetime;

        tzset();

        TSimpleTM::New(-1); // should not die here

        UNIT_ASSERT_VALUES_EQUAL((ui32)0, (ui32)TSimpleTM::New(0));
        UNIT_ASSERT((ui32)TSimpleTM::New(0).IsUTC());
        time_t t = time(nullptr);

        {
            struct tm tmt;
            Zero(tmt);
            gmtime_r(&t, &tmt);
            UNIT_ASSERT_VALUES_EQUAL_C((i64)t, (i64)TSimpleTM::New(t).AsTimeT(), ToString(t));   // time_t ->   gmt tm -> time_t
            UNIT_ASSERT_VALUES_EQUAL_C((i64)t, (i64)TSimpleTM::New(tmt).AsTimeT(), ToString(t)); // gmt tm -> time_t
            AssertStructTmEqual(PrintMarker("UTC:time_t", __LINE__, t),
                                tmt, TSimpleTM::New(t));
            AssertStructTmEqual(PrintMarker("UTC:tm", __LINE__, t),
                                tmt, TSimpleTM::New(tmt));
            UNIT_ASSERT(TSimpleTM::New(t).IsUTC());
            UNIT_ASSERT(TSimpleTM::New(tmt).IsUTC());
        }

        {
            struct tm tmt;
            Zero(tmt);
            localtime_r(&t, &tmt);

            UNIT_ASSERT_VALUES_EQUAL((i64)t, (i64)TSimpleTM::NewLocal(t).AsTimeT()); // time_t -> local tm -> time_t
            UNIT_ASSERT_VALUES_EQUAL((i64)t, (i64)TSimpleTM::New(tmt).AsTimeT());
            AssertStructTmEqual(PrintMarker("local:time_t", __LINE__, t),
                                tmt, TSimpleTM::NewLocal(t));
            AssertStructTmEqual(PrintMarker("local:tm", __LINE__, t),
                                tmt, TSimpleTM::New(tmt));
            AssertStructTmEqual(PrintMarker("local:tm:RegenerateFields", __LINE__, t),
                                tmt, TSimpleTM::New(tmt).RegenerateFields());
            AssertStructTmEqual(PrintMarker("local:time_t:SetRealDate", __LINE__, t),
                                tmt, TSimpleTM::NewLocal(t).SetRealDate(tmt.tm_year + 1900, tmt.tm_mon + 1, tmt.tm_mday, tmt.tm_hour, tmt.tm_min, tmt.tm_sec, tmt.tm_isdst));
        }

        {
            TSimpleTM tt = TSimpleTM::New(0);

            tt.SetRealDate(2012, 3, 30, 5, 6, 7);
            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1333083967, 2012, 3, 30, 5, 6, 7);

            tt.SetRealDate(2012, 3, 8, 5, 6, 7);
            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1331183167, 2012, 3, 8, 5, 6, 7);

            tt.SetRealDate(2010, 10, 4, 5, 6, 7);
            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1286168767, 2010, 10, 4, 5, 6, 7);

            tt.Add(TSimpleTM::F_MON);
            AssertSimpleTM(PrintMarker("UTC:AddMonth", __LINE__),
                           tt, 1288847167, 2010, 11, 4, 5, 6, 7);

            tt.Add(TSimpleTM::F_DAY);
            AssertSimpleTM(PrintMarker("UTC:AddDay", __LINE__),
                           tt, 1288933567, 2010, 11, 5, 5, 6, 7);

            tt.Add(TSimpleTM::F_YEAR);
            AssertSimpleTM(PrintMarker("UTC:AddYear", __LINE__),
                           tt, 1320469567, 2011, 11, 5, 5, 6, 7);

            for (ui32 i = 0; i < 365; ++i) {
                tt.Add(TSimpleTM::F_DAY);
            }

            AssertSimpleTM(PrintMarker("UTC:365*AddDay", __LINE__),
                           tt, 1352005567, 2012, 11, 4, 5, 6, 7);

            tt.Add(TSimpleTM::F_MON, -10);
            AssertSimpleTM(PrintMarker("UTC:AddMonth(-10)", __LINE__),
                           tt, 1325653567, 2012, 1, 4, 5, 6, 7);

            tt.Add(TSimpleTM::F_HOUR, -24 * 4 - 6);
            AssertSimpleTM(PrintMarker("UTC:AddHour(-102)", __LINE__),
                           tt, 1325286367, 2011, 12, 30, 23, 6, 7);
        }

        {
            TSimpleTM tt = TSimpleTM::New();

            tt.SetRealDate(2012, 2, 29);

            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1330473600, 2012, 2, 29, 0, 0, 0);

            tt.SetRealDate(2012, 2, 29);

            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1330473600, 2012, 2, 29, 0, 0, 0);

            tt.SetRealDate(2013, 12, 28);

            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1388188800, 2013, 12, 28, 0, 0, 0);

            tt.SetRealDate(2012, 10, 23);

            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1350950400, 2012, 10, 23, 0, 0, 0);

            tt.SetRealDate(2013, 3, 16);

            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1363392000, 2013, 3, 16, 0, 0, 0);

            tt.SetRealDate(2013, 2, 17);

            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1361059200, 2013, 2, 17, 0, 0, 0);

            tt.SetRealDate(2012, 12, 23);

            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1356220800, 2012, 12, 23, 0, 0, 0);

            tt.SetRealDate(2012, 5, 17);

            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1337212800, 2012, 5, 17, 0, 0, 0);

            tt.SetRealDate(2012, 6, 15);

            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1339718400, 2012, 6, 15, 0, 0, 0);

            tt.SetRealDate(2009, 3, 17);

            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1237248000, 2009, 3, 17, 0, 0, 0);

            tt.SetRealDate(2013, 8, 12);

            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1376265600, 2013, 8, 12, 0, 0, 0);

            tt.SetRealDate(2015, 12, 11, 10, 9, 8);

            AssertSimpleTM(PrintMarker("UTC:SetRealDate", __LINE__),
                           tt, 1449828548, 2015, 12, 11, 10, 9, 8);
        }
    }
} // Y_UNIT_TEST_SUITE(TSimpleTMTest)
