#include "format.h"

#include <library/cpp/testing/unittest/registar.h>
#include <util/charset/wide.h>

Y_UNIT_TEST_SUITE(TOutputStreamFormattingTest) {
    Y_UNIT_TEST(TestLeftPad) {
        TStringStream ss;
        ss << LeftPad(10, 4, '0');
        UNIT_ASSERT_VALUES_EQUAL("0010", ss.Str());

        ss.Clear();
        ss << LeftPad(222, 1);
        UNIT_ASSERT_VALUES_EQUAL("222", ss.Str());
    }

    Y_UNIT_TEST(TestRightPad) {
        TStringStream ss;
        ss << RightPad("aa", 4);
        UNIT_ASSERT_VALUES_EQUAL("aa  ", ss.Str());

        ss.Clear();
        ss << RightPad("aa", 1);
        UNIT_ASSERT_VALUES_EQUAL("aa", ss.Str());
    }

    Y_UNIT_TEST(TestTime) {
        TStringStream ss;

        ss << "[" << Time << "] "
           << "qwqw" << TimeHumanReadable << Endl;
    }

    Y_UNIT_TEST(TestHexReference) {
        /*
            One possible implementation of Hex() stores a reference to the given object.
            This can lead to wrong results if the given object is a temporary
            which is valid only during constructor call. The following code tries to
            demonstrate this. If the implementation stores a reference,
            the test fails if compiled with g++44 in debug build
            (without optimizations), but performs correctly in release build.
        */
        THolder<TStringStream> ss(new TStringStream);
        THolder<int> ii(new int(0x1234567));
        (*ss) << Hex(*ii);
        UNIT_ASSERT_VALUES_EQUAL("0x01234567", ss->Str());
    }

    Y_UNIT_TEST(TestHexText) {
        {
            TStringStream ss;
            ss << HexText(TStringBuf("abcи"));
            UNIT_ASSERT_VALUES_EQUAL("61 62 63 D0 B8", ss.Str());
        }
        {
            TStringStream ss;
            TUtf16String w = UTF8ToWide("abcи");
            ss << HexText<wchar16>(w);
            UNIT_ASSERT_VALUES_EQUAL("0061 0062 0063 0438", ss.Str());
        }
    }

    Y_UNIT_TEST(TestBin) {
        UNIT_ASSERT_VALUES_EQUAL(ToString(Bin(static_cast<ui32>(2), nullptr)), "10");
        UNIT_ASSERT_VALUES_EQUAL(ToString(SBin(static_cast<i32>(-2), nullptr)), "-10");
        UNIT_ASSERT_VALUES_EQUAL(ToString(SBin(static_cast<i32>(-2))), "-0b00000000000000000000000000000010");
        UNIT_ASSERT_VALUES_EQUAL(ToString(SBin(static_cast<i32>(-2), HF_FULL)), "-00000000000000000000000000000010");
        UNIT_ASSERT_VALUES_EQUAL(ToString(Bin(static_cast<ui32>(15), nullptr)), "1111");
        UNIT_ASSERT_VALUES_EQUAL(ToString(Bin(static_cast<ui32>(1))), "0b00000000000000000000000000000001");
        UNIT_ASSERT_VALUES_EQUAL(ToString(Bin(static_cast<ui32>(-1))), "0b11111111111111111111111111111111");
        UNIT_ASSERT_VALUES_EQUAL(ToString(Bin(static_cast<i32>(-1))), "0b11111111111111111111111111111111");
        UNIT_ASSERT_VALUES_EQUAL(ToString(Bin(static_cast<i32>(-1), nullptr)), "11111111111111111111111111111111");
        UNIT_ASSERT_VALUES_EQUAL(ToString(Bin(static_cast<ui32>(256))), "0b00000000000000000000000100000000");
        UNIT_ASSERT_VALUES_EQUAL(ToString(Bin(static_cast<ui8>(16))), "0b00010000");
        UNIT_ASSERT_VALUES_EQUAL(ToString(Bin(static_cast<ui64>(1234587912357ull))), "0b0000000000000000000000010001111101110011001011001000100010100101");
    }

    Y_UNIT_TEST(TestBinText) {
        UNIT_ASSERT_VALUES_EQUAL(ToString(BinText(TStringBuf("\1"))), "00000001");
        UNIT_ASSERT_VALUES_EQUAL(ToString(BinText(TStringBuf("\1\1"))), "00000001 00000001");
        UNIT_ASSERT_VALUES_EQUAL(ToString(BinText(TStringBuf("aaa"))), "01100001 01100001 01100001");
    }

    Y_UNIT_TEST(TestPrec) {
        TStringStream ss;
        ss << Prec(1.2345678901234567, PREC_AUTO);
        UNIT_ASSERT_VALUES_EQUAL("1.2345678901234567", ss.Str());

        ss.Clear();
        ss << Prec(1.2345678901234567, 3);
        UNIT_ASSERT_VALUES_EQUAL("1.23", ss.Str());

        ss.Clear();
        ss << Prec(1.2345678901234567, PREC_POINT_DIGITS, 3);
        UNIT_ASSERT_VALUES_EQUAL("1.235", ss.Str());
    }

    Y_UNIT_TEST(TestHumanReadableSize1000) {
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(0, SF_QUANTITY)), "0");

        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1, SF_QUANTITY)), "1");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1000, SF_QUANTITY)), "1K");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1234567, SF_QUANTITY)), "1.23M");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(12345678, SF_QUANTITY)), "12.3M");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(12345678 * 1000ull, SF_QUANTITY)), "12.3G");

        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1, SF_QUANTITY)), "-1");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1000, SF_QUANTITY)), "-1K");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1234567, SF_QUANTITY)), "-1.23M");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-12345678, SF_QUANTITY)), "-12.3M");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-12345678 * 1000ll, SF_QUANTITY)), "-12.3G");
    }

    Y_UNIT_TEST(TestHumanReadableSize1024) {
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(0, SF_BYTES)), "0B");

        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(100, SF_BYTES)), "100B");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1024, SF_BYTES)), "1KiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(2.25 * 1024 * 1024, SF_BYTES)), "2.25MiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(2.5 * 1024, SF_BYTES)), "2.5KiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(45.3 * 1024, SF_BYTES)), "45.3KiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1024 * 1024, SF_BYTES)), "1MiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(5 * 1024 * 1024, SF_BYTES)), "5MiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1236 * 1024 * 1024, SF_BYTES)), "1.21GiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1024ull * 1024 * 1024 * 1024, SF_BYTES)), "1TiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(100 / 3., SF_BYTES)), "33.3B");

        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-100, SF_BYTES)), "-100B");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1024, SF_BYTES)), "-1KiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-2.25 * 1024 * 1024, SF_BYTES)), "-2.25MiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-2.5 * 1024, SF_BYTES)), "-2.5KiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-45.3 * 1024, SF_BYTES)), "-45.3KiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1024 * 1024, SF_BYTES)), "-1MiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-5 * 1024 * 1024, SF_BYTES)), "-5MiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1236 * 1024 * 1024, SF_BYTES)), "-1.21GiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1024ll * 1024 * 1024 * 1024, SF_BYTES)), "-1TiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-100 / 3., SF_BYTES)), "-33.3B");

        // XXX: For 1000 <= x < 1024, Prec(x, 3) falls back to exponential form

        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1000ll, SF_BYTES)), "1000B");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1010ll, SF_BYTES)), "1010B");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1000ll * 1024, SF_BYTES)), "1000KiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1010ll * 1024, SF_BYTES)), "1010KiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1000ll * 1024 * 1024, SF_BYTES)), "1000MiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1010ll * 1024 * 1024, SF_BYTES)), "1010MiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1000ll * 1024 * 1024 * 1024, SF_BYTES)), "1000GiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1010ll * 1024 * 1024 * 1024, SF_BYTES)), "1010GiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1000ll * 1024 * 1024 * 1024 * 1024, SF_BYTES)), "1000TiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(1010ll * 1024 * 1024 * 1024 * 1024, SF_BYTES)), "1010TiB");

        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1000ll, SF_BYTES)), "-1000B");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1010ll, SF_BYTES)), "-1010B");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1000ll * 1024, SF_BYTES)), "-1000KiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1010ll * 1024, SF_BYTES)), "-1010KiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1000ll * 1024 * 1024, SF_BYTES)), "-1000MiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1010ll * 1024 * 1024, SF_BYTES)), "-1010MiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1000ll * 1024 * 1024 * 1024, SF_BYTES)), "-1000GiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1010ll * 1024 * 1024 * 1024, SF_BYTES)), "-1010GiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1000ll * 1024 * 1024 * 1024 * 1024, SF_BYTES)), "-1000TiB");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadableSize(-1010ll * 1024 * 1024 * 1024 * 1024, SF_BYTES)), "-1010TiB");
    }

    Y_UNIT_TEST(TestHumanReadableDuration) {
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::MicroSeconds(0))), "0us");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::MicroSeconds(1))), "1us");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::MicroSeconds(100))), "100us");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::MicroSeconds(1234))), "1.23ms");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::MicroSeconds(12345))), "12.3ms");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::MicroSeconds(1234567))), "1.23s");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::Seconds(5))), "5s");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::Seconds(59.9))), "59.9s");

        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::Seconds(60))), "1m");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::Seconds(61))), "1m 1s");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::Seconds(72))), "1m 12s");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::Seconds(620))), "10m 20s");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::Seconds(3600))), "1h");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::Seconds(3672))), "1h 1m 12s");
        UNIT_ASSERT_VALUES_EQUAL(ToString(HumanReadable(TDuration::Seconds(4220))), "1h 10m 20s");
    }
} // Y_UNIT_TEST_SUITE(TOutputStreamFormattingTest)
