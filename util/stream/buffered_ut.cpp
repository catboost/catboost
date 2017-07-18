#include "buffered.h"

#include <library/unittest/registar.h>

#include <util/generic/string.h>
#include <util/random/mersenne.h>

SIMPLE_UNIT_TEST_SUITE(TestBufferedIO) {
    template <class TOut>
    inline void Run(TOut && out) {
        TMersenne<ui64> r;

        for (size_t i = 0; i < 1000; ++i) {
            const size_t c = r.GenRand() % 10000;
            TString s;

            for (size_t j = 0; j < c; ++j) {
                s.append('A' + (r.GenRand() % 10));
            }

            out.Write(~s, +s);
        }
    }

    SIMPLE_UNIT_TEST(TestEqual) {
        TString s1;
        TString s2;

        Run(TBuffered<TStringOutput>(8000, s1));
        Run(TAdaptivelyBuffered<TStringOutput>(s2));

        UNIT_ASSERT_VALUES_EQUAL(s1, s2);
    }

    SIMPLE_UNIT_TEST(Test1) {
        TString s;

        TBuffered<TStringOutput>(100, s).Write("1", 1);

        UNIT_ASSERT_VALUES_EQUAL(s, "1");
    }

    SIMPLE_UNIT_TEST(Test2) {
        TString s;

        TBuffered<TStringOutput>(1, s).Write("12", 2);

        UNIT_ASSERT_VALUES_EQUAL(s, "12");
    }

    SIMPLE_UNIT_TEST(Test3) {
        TString s;

        auto&& b = TBuffered<TStringOutput>(1, s);

        b.Write("1", 1);
        b.Write("12", 2);

        UNIT_ASSERT_VALUES_EQUAL(s, "112");
    }

    SIMPLE_UNIT_TEST(TestInput) {
        TString s("0123456789abcdefghijklmn");
        TBuffered<TStringInput> in(5, s);
        char c;
        UNIT_ASSERT_VALUES_EQUAL(in.Read(&c, 1), 1); //1
        UNIT_ASSERT_VALUES_EQUAL(c, '0');
        UNIT_ASSERT_VALUES_EQUAL(in.Skip(4), 4);     //5 end of buffer
        UNIT_ASSERT_VALUES_EQUAL(in.Read(&c, 1), 1); //6
        UNIT_ASSERT_VALUES_EQUAL(c, '5');
        UNIT_ASSERT_VALUES_EQUAL(in.Skip(3), 3);     //9
        UNIT_ASSERT_VALUES_EQUAL(in.Read(&c, 1), 1); //10 end of buffer
        UNIT_ASSERT_VALUES_EQUAL(c, '9');
        UNIT_ASSERT_VALUES_EQUAL(in.Skip(3), 3);     //13
        UNIT_ASSERT_VALUES_EQUAL(in.Read(&c, 1), 1); //14 start new buffer
        UNIT_ASSERT_VALUES_EQUAL(c, 'd');
        UNIT_ASSERT_VALUES_EQUAL(in.Skip(6), 6);     //20
        UNIT_ASSERT_VALUES_EQUAL(in.Read(&c, 1), 1); //21 start new buffer
        UNIT_ASSERT_VALUES_EQUAL(c, 'k');
        UNIT_ASSERT_VALUES_EQUAL(in.Skip(6), 3); //24 eof
    }

    SIMPLE_UNIT_TEST(TestReadTo) {
        TString s("0123456789abc");
        TBuffered<TStringInput> in(2, s);
        TString t;
        UNIT_ASSERT_VALUES_EQUAL(in.ReadTo(t, '7'), 8);
        UNIT_ASSERT_VALUES_EQUAL(t, "0123456");
        UNIT_ASSERT_VALUES_EQUAL(in.ReadTo(t, '8'), 1);
        UNIT_ASSERT_VALUES_EQUAL(t, "");
        UNIT_ASSERT_VALUES_EQUAL(in.ReadTo(t, 'z'), 4);
        UNIT_ASSERT_VALUES_EQUAL(t, "9abc");
    }
}
