#include "base.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/utility.h>
#include <util/generic/ylimits.h>
#include <util/generic/ymath.h>
#include <util/string/cast.h>
#include <util/stream/output.h>
#include <util/system/compat.h>
#include <util/random/random.h>

#include <climits>

using namespace std::chrono_literals;

struct TTestTime {
    const time_t T_ = 987654321;
    const char* Date_ = "Thu Apr 19 04:25:21 2001\n";
    const char* SprintDate_ = "20010419";
    const long SprintSecs_ = 15921;
    struct tm Tm_;
    TTestTime() {
        Tm_.tm_sec = 21;
        Tm_.tm_min = 25;
        Tm_.tm_hour = 4;
        Tm_.tm_mday = 19;
        Tm_.tm_mon = 3;
        Tm_.tm_year = 101;
        Tm_.tm_wday = 4;
        Tm_.tm_yday = 108;
    }
};

namespace {
    inline void OldDate8601(char* buf, size_t bufLen, time_t when) {
        struct tm theTm;
        struct tm* ret = nullptr;

        ret = GmTimeR(&when, &theTm);

        if (ret) {
            snprintf(buf, bufLen, "%04d-%02d-%02dT%02d:%02d:%02dZ", theTm.tm_year + 1900, theTm.tm_mon + 1, theTm.tm_mday, theTm.tm_hour, theTm.tm_min, theTm.tm_sec);
        } else {
            *buf = '\0';
        }
    }
} // namespace

Y_UNIT_TEST_SUITE(TestSprintDate) {
    Y_UNIT_TEST(Year9999) {
        struct tm t;
        t.tm_year = 9999 - 1900;
        t.tm_mday = 1;
        t.tm_mon = 10;

        char buf[DATE_BUF_LEN];
        DateToString(buf, t);

        TString expectedDate = "99991101";

        UNIT_ASSERT_VALUES_EQUAL(expectedDate, ToString(buf));
    }
    Y_UNIT_TEST(YearAfter9999) {
        struct tm t;
        t.tm_year = 123456 - 1900;
        t.tm_mday = 1;
        t.tm_mon = 10;

        char buf[DATE_BUF_LEN];
        UNIT_ASSERT_EXCEPTION(DateToString(buf, t), yexception);
    }
    Y_UNIT_TEST(SmallYear) {
        struct tm t;
        t.tm_year = 0 - 1900;
        t.tm_mday = 1;
        t.tm_mon = 10;

        char buf[DATE_BUF_LEN];
        DateToString(buf, t);

        const TString expectedDate = TString("00001101");

        UNIT_ASSERT_VALUES_EQUAL(expectedDate, ToString(buf));
    }
    Y_UNIT_TEST(SmallYearAndMonth) {
        struct tm t;
        t.tm_year = 99 - 1900;
        t.tm_mday = 1;
        t.tm_mon = 0;

        char buf[DATE_BUF_LEN];
        DateToString(buf, t);

        const TString expectedDate = TString("00990101");

        UNIT_ASSERT_VALUES_EQUAL(expectedDate, ToString(buf));
    }
    Y_UNIT_TEST(FromZeroTimestamp) {
        const time_t timestamp = 0;

        char buf[DATE_BUF_LEN];
        DateToString(buf, timestamp);

        const TString expectedDate = TString("19700101");

        UNIT_ASSERT_VALUES_EQUAL(expectedDate, ToString(buf));
    }
    Y_UNIT_TEST(FromTimestamp) {
        const time_t timestamp = 1524817858;

        char buf[DATE_BUF_LEN];
        DateToString(buf, timestamp);

        const TString expectedDate = TString("20180427");

        UNIT_ASSERT_VALUES_EQUAL(expectedDate, ToString(buf));
    }
    Y_UNIT_TEST(FromTimestampAsTString) {
        const time_t timestamp = 1524817858;

        const TString expectedDate = TString("20180427");

        UNIT_ASSERT_VALUES_EQUAL(expectedDate, DateToString(timestamp));
    }
    Y_UNIT_TEST(YearToString) {
        struct tm t;
        t.tm_year = 99 - 1900;
        t.tm_mday = 1;
        t.tm_mon = 0;

        TString expectedYear = TString("0099");

        UNIT_ASSERT_VALUES_EQUAL(expectedYear, YearToString(t));
    }
    Y_UNIT_TEST(YearToStringBigYear) {
        struct tm t;
        t.tm_year = 123456 - 1900;
        t.tm_mday = 1;
        t.tm_mon = 0;

        UNIT_ASSERT_EXCEPTION(YearToString(t), yexception);
    }
    Y_UNIT_TEST(YearToStringAsTimestamp) {
        const time_t timestamp = 1524817858;

        const TString expectedYear = TString("2018");

        UNIT_ASSERT_VALUES_EQUAL(expectedYear, YearToString(timestamp));
    }
} // Y_UNIT_TEST_SUITE(TestSprintDate)

Y_UNIT_TEST_SUITE(TDateTimeTest) {
    Y_UNIT_TEST(Test8601) {
        char buf1[100];
        char buf2[100];

        for (size_t i = 0; i < 1000000; ++i) {
            const time_t t = RandomNumber<ui32>();

            OldDate8601(buf1, sizeof(buf1), t);
            sprint_date8601(buf2, t);

            UNIT_ASSERT_VALUES_EQUAL(TStringBuf(buf1), TStringBuf(buf2));
        }
    }

    inline bool CompareTM(const struct tm& a, const struct tm& b) {
        return (
            a.tm_sec == b.tm_sec &&
            a.tm_min == b.tm_min &&
            a.tm_hour == b.tm_hour &&
            a.tm_mday == b.tm_mday &&
            a.tm_mon == b.tm_mon &&
            a.tm_year == b.tm_year &&
            a.tm_wday == b.tm_wday &&
            a.tm_yday == b.tm_yday);
    }

    static inline TString Str(const struct tm& a) {
        return TStringBuilder() << "("
                                << a.tm_sec << ", "
                                << a.tm_min << ", "
                                << a.tm_hour << ", "
                                << a.tm_mday << ", "
                                << a.tm_mon << ", "
                                << a.tm_year << ", "
                                << a.tm_wday << ", "
#if !defined(_musl_) && !defined(_win_)
                                << a.tm_yday
#endif
                                << ")";
    }

    Y_UNIT_TEST(TestBasicFuncs) {
        ui64 mlsecB = millisec();
        ui64 mcrsecB = MicroSeconds();
        struct timeval tvB;
        gettimeofday(&tvB, nullptr);

        usleep(100000);

        ui64 mlsecA = millisec();
        ui64 mcrsecA = MicroSeconds();
        struct timeval tvA;
        gettimeofday(&tvA, nullptr);

        UNIT_ASSERT(mlsecB + 90 < mlsecA);
        UNIT_ASSERT((mcrsecB + 90000 < mcrsecA));
        // UNIT_ASSERT(ToMicroSeconds(&tvB) + 90000 < ToMicroSeconds(&tvA));
        // UNIT_ASSERT(TVdiff(tvB, tvA) == long(ToMicroSeconds(&tvA) - ToMicroSeconds(&tvB)));
    }

    Y_UNIT_TEST(TestCompatFuncs) {
        struct tm t;
        struct tm* tret = nullptr;
        TTestTime e;
        tret = gmtime_r(&e.T_, &t);
        UNIT_ASSERT(tret == &t);
        UNIT_ASSERT(CompareTM(e.Tm_, t));

        /*
         * strptime seems to be broken on Mac OS X:
         *
         *   struct tm t;
         *   char *ret = strptime("Jul", "%b ", &t);
         *   printf("-%s-\n", ret);
         *
         * yields "- -": ret contains a pointer to a substring of the format string,
         * that should never occur: function returns either NULL or pointer to buf substring.
         *
         * So this test fails on Mac OS X.
         */

        struct tm t2;
        Zero(t2);
        char* ret = strptime(e.Date_, "%a %b %d %H:%M:%S %Y\n ", &t2);
        UNIT_ASSERT(ret == e.Date_ + strlen(e.Date_));
        UNIT_ASSERT_VALUES_EQUAL(Str(e.Tm_), Str(t2));
        time_t t3 = timegm(&t);
        UNIT_ASSERT(t3 == e.T_);
    }

    Y_UNIT_TEST(TestSprintSscan) {
        char buf[256];
        long secs;
        TTestTime e;

        sprint_gm_date(buf, e.T_, &secs);
        UNIT_ASSERT(strcmp(buf, e.SprintDate_) == 0);
        UNIT_ASSERT(secs == e.SprintSecs_);

        struct tm t;
        Zero(t);
        bool ret = sscan_date(buf, t);
        UNIT_ASSERT(ret);
        UNIT_ASSERT(
            t.tm_year == e.Tm_.tm_year &&
            t.tm_mon == e.Tm_.tm_mon &&
            t.tm_mday == e.Tm_.tm_mday);

        buf[0] = 'a';
        ret = sscan_date(buf, t);
        UNIT_ASSERT(!ret);
    }

    Y_UNIT_TEST(TestNow) {
        i64 seconds = Seconds();
        i64 milliseconds = millisec();
        i64 microseconds = MicroSeconds();
        UNIT_ASSERT(Abs(seconds - milliseconds / 1000) <= 1);
        UNIT_ASSERT(Abs(milliseconds - microseconds / 1000) < 100);
        UNIT_ASSERT(seconds > 1243008607); // > time when test was written
    }

    Y_UNIT_TEST(TestStrftime) {
        struct tm tm;
        Zero(tm);
        tm.tm_year = 109;
        tm.tm_mon = 4;
        tm.tm_mday = 29;
        UNIT_ASSERT_STRINGS_EQUAL("2009-05-29", Strftime("%Y-%m-%d", &tm));
    }

    Y_UNIT_TEST(TestNanoSleep) {
        NanoSleep(0);
        NanoSleep(1);
        NanoSleep(1000);
        NanoSleep(1000000);
    }

    static bool TimeZoneEq(const char* zone0, const char* zone1) {
        if (strcmp(zone0, "GMT") == 0) {
            zone0 = "UTC";
        }
        if (strcmp(zone1, "GMT") == 0) {
            zone1 = "UTC";
        }
        return strcmp(zone0, zone1) == 0;
    }

    static bool CompareTMFull(const tm* t0, const tm* t1) {
        return t0 && t1 &&
               CompareTM(*t0, *t1) &&
               (t0->tm_isdst == t1->tm_isdst)
#ifndef _win_
               && (t0->tm_gmtoff == t1->tm_gmtoff) &&
               TimeZoneEq(t0->tm_zone, t1->tm_zone)
#endif // _win_
               && true;
    }

    void TestGmTimeR(time_t starttime, time_t finishtime, int steps) {
        time_t step = (finishtime - starttime) / steps;
        struct tm tms0, tms1;
        struct tm* ptm0 = nullptr;
        struct tm* ptm1 = nullptr;
        for (time_t t = starttime; t < finishtime; t += step) {
            ptm0 = GmTimeR(&t, &tms0);
            UNIT_ASSERT_EQUAL(ptm0, &tms0);

#ifdef _win_
            if (tms0.tm_year + 1900 > 3000) {
                // Windows: _MAX__TIME64_T == 23:59:59. 12/31/3000 UTC
                continue;
            }
#endif

            ptm1 = gmtime_r(&t, &tms1);
            if (!ptm1) {
                continue;
            }
            UNIT_ASSERT_EQUAL(ptm1, &tms1);
            UNIT_ASSERT(CompareTMFull(ptm0, ptm1));
        }
    }

    Y_UNIT_TEST(TestGmTimeRLongRange) {
        time_t starttime = static_cast<time_t>(-86397839500LL); // 29-Jan-2668 B.C.
        time_t finishtime = static_cast<time_t>(0xFFFFFFFF * 20);
        TestGmTimeR(starttime, finishtime, 101);
    }

    Y_UNIT_TEST(TestGmTimeRNowdays) {
        time_t starttime = static_cast<time_t>(0);             // 1970
        time_t finishtime = static_cast<time_t>(6307200000LL); // 2170
        TestGmTimeR(starttime, finishtime, 303);
    }
} // Y_UNIT_TEST_SUITE(TDateTimeTest)

Y_UNIT_TEST_SUITE(DateTimeTest) {
    Y_UNIT_TEST(TestDurationFromFloat) {
        UNIT_ASSERT_EQUAL(TDuration::MilliSeconds(500), TDuration::Seconds(0.5));
        UNIT_ASSERT_EQUAL(TDuration::MilliSeconds(500), TDuration::Seconds(0.5f));
    }

    Y_UNIT_TEST(TestSecondsLargeValue) {
        unsigned int seconds = UINT_MAX;
        UNIT_ASSERT_VALUES_EQUAL(((ui64)seconds) * 1000000, TDuration::Seconds(seconds).MicroSeconds());
    }

    Y_UNIT_TEST(TestToString) {
#define CHECK_CONVERTIBLE(v)                                         \
    do {                                                             \
        UNIT_ASSERT_VALUES_EQUAL(v, ToString(TDuration::Parse(v)));  \
        UNIT_ASSERT_VALUES_EQUAL(v, TDuration::Parse(v).ToString()); \
    } while (0)
#if 0

        CHECK_CONVERTIBLE("10s");
        CHECK_CONVERTIBLE("1234s");
        CHECK_CONVERTIBLE("1234ms");
        CHECK_CONVERTIBLE("12ms");
        CHECK_CONVERTIBLE("12us");
        CHECK_CONVERTIBLE("1234us");
#endif

        CHECK_CONVERTIBLE("1.000000s");
        CHECK_CONVERTIBLE("11234.000000s");
        CHECK_CONVERTIBLE("0.011122s");
        CHECK_CONVERTIBLE("33.011122s");
    }

    Y_UNIT_TEST(TestFromString) {
        static const struct T {
            const char* const Str;
            const TDuration::TValue MicroSeconds;
            const bool Parseable;
        } tests[] = {
            {"0", 0, true},
            {"1", 1000000, true},
            {"2s", 2000000, true},
            {"3ms", 3000, true},
            {"x3ms", 0, false},
        };

        for (const T* t = tests; t != std::end(tests); ++t) {
            // FromString
            bool parsed = false;
            try {
                TDuration time = FromString<TDuration>(t->Str);
                parsed = true;
                UNIT_ASSERT_EQUAL(t->MicroSeconds, time.MicroSeconds());
            } catch (const yexception&) {
                UNIT_ASSERT_VALUES_EQUAL(parsed, t->Parseable);
            }
            // TryFromString
            TDuration tryTime;
            UNIT_ASSERT_VALUES_EQUAL(TryFromString<TDuration>(t->Str, tryTime), t->Parseable);
            if (t->Parseable) {
                UNIT_ASSERT_EQUAL(t->MicroSeconds, tryTime.MicroSeconds());
            }
        }
    }

    Y_UNIT_TEST(TestSleep) {
        // check does not throw
        Sleep(TDuration::Seconds(0));
        Sleep(TDuration::MicroSeconds(1));
        Sleep(TDuration::MilliSeconds(1));
    }

    Y_UNIT_TEST(TestInstantToString) {
        UNIT_ASSERT_VALUES_EQUAL(TString("2009-08-06T15:19:06.023455Z"), ToString(TInstant::Seconds(1249571946) + TDuration::MicroSeconds(23455)));
        UNIT_ASSERT_VALUES_EQUAL(TString("2022-08-23T13:04:43.023455Z"), ToString(TInstant::Seconds(1661259883) + TDuration::MicroSeconds(23455)));
        UNIT_ASSERT_VALUES_EQUAL(TString("2122-11-23T15:12:26.023455Z"), ToString(TInstant::Seconds(4824889946) + TDuration::MicroSeconds(23455)));
        UNIT_ASSERT_VALUES_EQUAL(TString("2009-08-06T15:19:06.023455Z"), (TInstant::Seconds(1249571946) + TDuration::MicroSeconds(23455)).ToString());
        UNIT_ASSERT_VALUES_EQUAL(TString("2009-08-06T15:19:06Z"), (TInstant::Seconds(1249571946) + TDuration::MicroSeconds(23455)).ToStringUpToSeconds());
    }

    Y_UNIT_TEST(TestInstantToRfc822String) {
        UNIT_ASSERT_VALUES_EQUAL(TString("Thu, 06 Aug 2009 15:19:06 GMT"), (TInstant::Seconds(1249571946) + TDuration::MicroSeconds(23455)).ToRfc822String());
    }

    Y_UNIT_TEST(TestInstantMath) {
        UNIT_ASSERT_VALUES_EQUAL(TInstant::Seconds(1719), TInstant::Seconds(1700) + TDuration::Seconds(19));
        // overflow
        UNIT_ASSERT_VALUES_EQUAL(TInstant::Max(), TInstant::Max() - TDuration::Seconds(10) + TDuration::Seconds(19));
        // underflow
        UNIT_ASSERT_VALUES_EQUAL(TInstant::Zero(), TInstant::Seconds(1000) - TDuration::Seconds(2000));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Zero(), TInstant::Seconds(1000) - TInstant::Seconds(2000));
    }

    Y_UNIT_TEST(TestDurationMath) {
        TDuration empty;
        UNIT_ASSERT(!empty);
        // ensure that this compiles too
        if (empty) {
            UNIT_ASSERT(false);
        }
        TDuration nonEmpty = TDuration::MicroSeconds(1);
        UNIT_ASSERT(nonEmpty);

        UNIT_ASSERT_VALUES_EQUAL(TDuration::Seconds(110), TDuration::Seconds(77) + TDuration::Seconds(33));
        // overflow
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Max(), TDuration::Max() - TDuration::Seconds(1) + TDuration::Seconds(10));
        // underflow
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Zero(), TDuration::Seconds(20) - TDuration::Seconds(200));
        // division
        UNIT_ASSERT_DOUBLES_EQUAL(TDuration::Minutes(1) / TDuration::Seconds(10), 6.0, 1e-9);
    }

    Y_UNIT_TEST(TestDurationGetters) {
        const TDuration value = TDuration::MicroSeconds(1234567);
        UNIT_ASSERT_VALUES_EQUAL(value.Seconds(), 1);
        UNIT_ASSERT_DOUBLES_EQUAL(value.SecondsFloat(), 1.234567, 1e-9);

        UNIT_ASSERT_VALUES_EQUAL(value.MilliSeconds(), 1234);
        UNIT_ASSERT_DOUBLES_EQUAL(value.MillisecondsFloat(), 1234.567, 1e-9);

        UNIT_ASSERT_VALUES_EQUAL(value.MicroSeconds(), 1234567);
    }

    template <class T>
    void TestTimeUnits() {
        T withTime = T::MicroSeconds(1249571946000000L);
        T onlyMinutes = T::MicroSeconds(1249571940000000L);
        T onlyHours = T::MicroSeconds(1249570800000000L);
        T onlyDays = T::MicroSeconds(1249516800000000L);
        ui64 minutes = 20826199;
        ui64 hours = 347103;
        ui64 days = 14462;

        UNIT_ASSERT_VALUES_EQUAL(withTime.Minutes(), minutes);
        UNIT_ASSERT_VALUES_EQUAL(onlyMinutes, T::Minutes(minutes));
        UNIT_ASSERT_VALUES_EQUAL(onlyMinutes.Minutes(), minutes);

        UNIT_ASSERT_VALUES_EQUAL(withTime.Hours(), hours);
        UNIT_ASSERT_VALUES_EQUAL(onlyMinutes.Hours(), hours);
        UNIT_ASSERT_VALUES_EQUAL(onlyHours, T::Hours(hours));
        UNIT_ASSERT_VALUES_EQUAL(onlyHours.Hours(), hours);

        UNIT_ASSERT_VALUES_EQUAL(withTime.Days(), days);
        UNIT_ASSERT_VALUES_EQUAL(onlyHours.Days(), days);
        UNIT_ASSERT_VALUES_EQUAL(onlyDays, T::Days(days));
        UNIT_ASSERT_VALUES_EQUAL(onlyDays.Days(), days);
    }

    Y_UNIT_TEST(TestInstantUnits) {
        TestTimeUnits<TInstant>();
    }

    Y_UNIT_TEST(TestDurationUnits) {
        TestTimeUnits<TDuration>();
    }

    Y_UNIT_TEST(TestNoexceptConstruction) {
        UNIT_ASSERT_EXCEPTION(TDuration::MilliSeconds(FromString(TStringBuf("not a number"))), yexception);
        UNIT_ASSERT_EXCEPTION(TDuration::Seconds(FromString(TStringBuf("not a number"))), yexception);
    }

    Y_UNIT_TEST(TestFromValueForTDuration) {
        // check that FromValue creates the same TDuration
        TDuration d1 = TDuration::MicroSeconds(12345);
        TDuration d2 = TDuration::FromValue(d1.GetValue());

        UNIT_ASSERT_VALUES_EQUAL(d1, d2);
    }

    Y_UNIT_TEST(TestFromValueForTInstant) {
        // check that FromValue creates the same TInstant
        TInstant i1 = TInstant::MicroSeconds(12345);
        TInstant i2 = TInstant::FromValue(i1.GetValue());

        UNIT_ASSERT_VALUES_EQUAL(i1, i2);
    }

    Y_UNIT_TEST(TestTimeGmDateConversion) {
        tm time{};
        time_t timestamp = 0;

        // Check all days till year 2106 (max year representable if time_t is 32 bit)
        while (time.tm_year < 2106 - 1900) {
            timestamp += 86400;

            GmTimeR(&timestamp, &time);
            time_t newTimestamp = TimeGM(&time);

            UNIT_ASSERT_VALUES_EQUAL_C(
                newTimestamp,
                timestamp,
                "incorrect date " << (1900 + time.tm_year) << "-" << (time.tm_mon + 1) << "-" << time.tm_mday);
        }
    }

    Y_UNIT_TEST(TestTDurationConstructorFromStdChronoDuration) {
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Zero(), TDuration(0ms));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::MicroSeconds(42), TDuration(42us));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::MicroSeconds(42000000000000L), TDuration(42000000000000us));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::MilliSeconds(42), TDuration(42ms));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::MilliSeconds(42.75), TDuration(42.75ms));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Seconds(42), TDuration(42s));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Seconds(42.25), TDuration(42.25s));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Minutes(42), TDuration(42min));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Hours(42), TDuration(42h));

        // TDuration doesn't support negative durations
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Zero(), TDuration(-5min));

        UNIT_ASSERT_VALUES_EQUAL(TDuration::MilliSeconds(5), TDuration(std::chrono::duration<i8, std::milli>{5ms}));

#if defined(_LIBCPP_STD_VER) && _LIBCPP_STD_VER > 17
        // libstdc++ does not provide std::chrono::days at the time
        // Consider removing this code upon OS_SDK update
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Days(1), TDuration(std::chrono::days{1}));
#endif

        // clump
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Zero(), TDuration(std::chrono::duration<i64>{-1}));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Zero(), TDuration(std::chrono::duration<double>{-1}));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Max(),
                                 TDuration(std::chrono::duration<ui64, std::ratio<3600>>{static_cast<ui64>(1e18)}));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Max(),
                                 TDuration(std::chrono::duration<i64, std::milli>{static_cast<i64>(::Max<ui64>() / 1000)}));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Max(),
                                 TDuration(std::chrono::duration<double, std::ratio<3600>>{1e18}));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Max(),
                                 TDuration(std::chrono::duration<double, std::milli>{::Max<ui64>() / 1000}));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Max(), TDuration(std::chrono::duration<double, std::milli>{
                                                       static_cast<double>(::Max<ui64>()) / 1000 + 0.1}));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Max(), TDuration(std::chrono::duration<float, std::milli>{
                                                       static_cast<float>(::Max<ui64>()) / 1000 + 0.1}));
    }

    Y_UNIT_TEST(TestTDurationCompareWithStdChronoDuration) {
        UNIT_ASSERT(TDuration::Zero() == 0ms);
        UNIT_ASSERT(TDuration::Seconds(42) == 42s);

        UNIT_ASSERT(0ms == TDuration::Zero());

        UNIT_ASSERT(TDuration::Zero() != 1ms);
        UNIT_ASSERT(TDuration::Zero() != -1ms);
        UNIT_ASSERT(TDuration::MilliSeconds(1) != -1ms);
        UNIT_ASSERT(TDuration::MilliSeconds(1) != -1ms);

        UNIT_ASSERT(1ms != TDuration::Zero());

        UNIT_ASSERT(TDuration::Seconds(2) < 3s);
        UNIT_ASSERT(3s > TDuration::Seconds(2));
        UNIT_ASSERT(!(TDuration::Seconds(2) < 1s));
        UNIT_ASSERT(!(TDuration::Seconds(2) < -3s));
        UNIT_ASSERT(!(TDuration::Seconds(2) < 2s));

        UNIT_ASSERT(2s < TDuration::Seconds(3));

        UNIT_ASSERT(TDuration::Seconds(2) <= 3s);
        UNIT_ASSERT(!(TDuration::Seconds(2) <= 1s));
        UNIT_ASSERT(!(TDuration::Seconds(2) <= -3s));
        UNIT_ASSERT(TDuration::Seconds(2) <= 2s);

        UNIT_ASSERT(2s <= TDuration::Seconds(2));

        UNIT_ASSERT(TDuration::Seconds(2) > -2s);
        UNIT_ASSERT(TDuration::Seconds(2) > 1s);
        UNIT_ASSERT(TDuration::Seconds(2) > 0s);
        UNIT_ASSERT(!(TDuration::Seconds(2) > 3s));
        UNIT_ASSERT(!(TDuration::Seconds(2) > 2s));

        UNIT_ASSERT(2s > TDuration::Seconds(1));

        UNIT_ASSERT(TDuration::Seconds(2) >= -2s);
        UNIT_ASSERT(TDuration::Seconds(2) >= 1s);
        UNIT_ASSERT(TDuration::Seconds(2) >= 0s);
        UNIT_ASSERT(!(TDuration::Seconds(2) >= 3s));
        UNIT_ASSERT(TDuration::Seconds(2) >= 2s);

        UNIT_ASSERT(2s >= TDuration::Seconds(2));

        static_assert(TDuration::Zero() == 0ms);
        static_assert(TDuration::Zero() < 1ms);
    }

    Y_UNIT_TEST(TestAdditionOfStdChronoDuration) {
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Seconds(1) + 2s, TDuration::Seconds(3));
        UNIT_ASSERT_VALUES_EQUAL(2s + TDuration::Seconds(1), TDuration::Seconds(3));
        UNIT_ASSERT_VALUES_EQUAL(-2s + TDuration::Seconds(3), TDuration::Seconds(1));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Seconds(3) + (-2s), TDuration::Seconds(1));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Seconds(3) - 2s, TDuration::Seconds(1));
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Seconds(1) - (-2s), TDuration::Seconds(3));
        UNIT_ASSERT_VALUES_EQUAL(3s - TDuration::Seconds(2), TDuration::Seconds(1));
        UNIT_ASSERT_VALUES_EQUAL(3s - TDuration::Seconds(4), TDuration::Zero());

        UNIT_ASSERT_VALUES_EQUAL(TInstant::Seconds(1) + 2s, TInstant::Seconds(3));
        UNIT_ASSERT_VALUES_EQUAL(TInstant::Seconds(3) + (-2s), TInstant::Seconds(1));
        UNIT_ASSERT_VALUES_EQUAL(TInstant::Seconds(3) - 2s, TInstant::Seconds(1));
        UNIT_ASSERT_VALUES_EQUAL(TInstant::Seconds(1) - (-2s), TInstant::Seconds(3));

        // Operations between TDuration/TInstant and std::chrono::duration are performed
        // with saturation according to the rules of TDuration/TInstant
        UNIT_ASSERT_VALUES_EQUAL(TDuration::Max() + 1h, TDuration::Max());
        UNIT_ASSERT_VALUES_EQUAL(TInstant::Max() + 1h, TInstant::Max());
        UNIT_ASSERT_VALUES_EQUAL(1h + TDuration::Max(), TDuration::Max());
        UNIT_ASSERT_VALUES_EQUAL(TInstant::Max() + 1h, TInstant::Max());

        UNIT_ASSERT_VALUES_EQUAL(TDuration::Max() - (-1h), TDuration::Max());
        UNIT_ASSERT_VALUES_EQUAL(TInstant::Max() - (-1h), TInstant::Max());
        UNIT_ASSERT_VALUES_EQUAL(TInstant::Max() - (-1h), TInstant::Max());

        UNIT_ASSERT_VALUES_EQUAL(-1h - TDuration::Max(), TDuration::Zero());
        UNIT_ASSERT_VALUES_EQUAL(1h - TDuration::Max(), TDuration::Zero());

        static_assert(TDuration::Zero() + 1s == 1s);
        static_assert(TInstant::Seconds(1) + 1s == TInstant::Seconds(2));
    }
} // Y_UNIT_TEST_SUITE(DateTimeTest)
