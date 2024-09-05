#include "length.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/string.h>

Y_UNIT_TEST_SUITE(TestLengthIO) {
    Y_UNIT_TEST(TestLengthLimitedInput) {
        char buf[16];

        TStringStream s1("abcd");
        TLengthLimitedInput l1(&s1, 2);
        UNIT_ASSERT_VALUES_EQUAL(l1.Load(buf, 3), 2);
        UNIT_ASSERT_VALUES_EQUAL(l1.Read(buf, 1), 0);
    }

    Y_UNIT_TEST(TestCountingInput) {
        char buf[16];

        TStringStream s1("abc\ndef\n");
        TCountingInput l1(&s1);

        TString s;
        l1.ReadLine(s);
        UNIT_ASSERT_VALUES_EQUAL(l1.Counter(), 4);

        l1.Load(buf, 1);
        UNIT_ASSERT_VALUES_EQUAL(l1.Counter(), 5);

        l1.Skip(1);
        UNIT_ASSERT_VALUES_EQUAL(l1.Counter(), 6);

        l1.ReadLine(s);
        UNIT_ASSERT_VALUES_EQUAL(l1.Counter(), 8);
    }

    Y_UNIT_TEST(TestCountingOutput) {
        TStringStream s1;
        TCountingOutput l1(&s1);

        l1.Write('1');
        UNIT_ASSERT_VALUES_EQUAL(l1.Counter(), 1);

        l1.Write(TString("abcd"));
        UNIT_ASSERT_VALUES_EQUAL(l1.Counter(), 5);

        TString buf("aaa");
        IOutputStream::TPart parts[] = {{buf.data(), buf.size()}, {buf.data(), buf.size()}, {buf.data(), buf.size()}};
        l1.Write(parts, 3);
        UNIT_ASSERT_VALUES_EQUAL(l1.Counter(), 14);
    }
} // Y_UNIT_TEST_SUITE(TestLengthIO)
