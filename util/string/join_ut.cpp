#include "join.h"

#include <library/cpp/testing/unittest/registar.h>
#include <util/generic/vector.h>

#include <util/stream/output.h>

struct TCustomData {
    TVector<int> Ints;
};

TString ToString(const TCustomData& d) {
    return JoinSeq("__", d.Ints);
}

Y_UNIT_TEST_SUITE(JoinStringTest) {
    Y_UNIT_TEST(ScalarItems) {
        UNIT_ASSERT_EQUAL(Join(',', 10, 11.1, "foobar"), "10,11.1,foobar");
        UNIT_ASSERT_EQUAL(Join(", ", 10, 11.1, "foobar"), "10, 11.1, foobar");
        UNIT_ASSERT_EQUAL(Join(", ", 10, 11.1, TString("foobar")), "10, 11.1, foobar");

        UNIT_ASSERT_EQUAL(Join('#', 0, "a", "foobar", -1.4, TStringBuf("aaa")), "0#a#foobar#-1.4#aaa");
        UNIT_ASSERT_EQUAL(Join("", "", ""), "");
        UNIT_ASSERT_EQUAL(Join("", "a", "b", "c"), "abc");
        UNIT_ASSERT_EQUAL(Join("", "a", "b", "", "c"), "abc");
        UNIT_ASSERT_EQUAL(Join(" ", "a", "b", "", "c"), "a b  c");
    }

    Y_UNIT_TEST(IntContainerItems) {
        int v[] = {1, 2, 3};
        TVector<int> vv(v, v + 3);
        UNIT_ASSERT_EQUAL(JoinSeq(" ", vv), "1 2 3");
        UNIT_ASSERT_EQUAL(JoinSeq(" ", vv), JoinRange(" ", vv.begin(), vv.end()));
        UNIT_ASSERT_EQUAL(JoinRange(" ", v, v + 2), "1 2");
        UNIT_ASSERT_EQUAL(JoinSeq(" ", {}), "");
        UNIT_ASSERT_EQUAL(JoinSeq(" ", {42}), "42");
        UNIT_ASSERT_EQUAL(JoinSeq(" ", {1, 2, 3}), "1 2 3");
        UNIT_ASSERT_VALUES_EQUAL(JoinSeq(" ", v), "1 2 3");
    }

    Y_UNIT_TEST(StrContainerItems) {
        // try various overloads and template type arguments
        static const char* const result = "1 22 333";
        static const char* const v[] = {"1", "22", "333"};
        TVector<const char*> vchar(v, v + sizeof(v) / sizeof(v[0]));
        TVector<TStringBuf> vbuf(v, v + sizeof(v) / sizeof(v[0]));
        TVector<TString> vstring(v, v + sizeof(v) / sizeof(v[0]));

        // ranges
        UNIT_ASSERT_EQUAL(JoinRange(" ", v, v + 3), result);
        UNIT_ASSERT_EQUAL(JoinRange(" ", vchar.begin(), vchar.end()), result);
        UNIT_ASSERT_EQUAL(JoinRange(" ", vbuf.begin(), vbuf.end()), result);
        UNIT_ASSERT_EQUAL(JoinRange(" ", vstring.begin(), vstring.end()), result);
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", v, v + 3);
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", vchar.begin(), vchar.end());
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", vbuf.begin(), vbuf.end());
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", vstring.begin(), vstring.end());
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }

        // vectors
        UNIT_ASSERT_EQUAL(JoinSeq(" ", vchar), result);
        UNIT_ASSERT_EQUAL(JoinSeq(" ", vbuf), result);
        UNIT_ASSERT_EQUAL(JoinSeq(" ", vstring), result);
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", vchar);
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", vbuf);
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", vstring);
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }

        // initializer lists with type deduction
        UNIT_ASSERT_EQUAL(JoinSeq(" ", {v[0], v[1], v[2]}), result);
        UNIT_ASSERT_EQUAL(JoinSeq(" ", {vchar[0], vchar[1], vchar[2]}), result);
        UNIT_ASSERT_EQUAL(JoinSeq(" ", {vbuf[0], vbuf[1], vbuf[2]}), result);
        UNIT_ASSERT_EQUAL(JoinSeq(" ", {vstring[0], vstring[1], vstring[2]}), result);
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", {v[0], v[1], v[2]});
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", {vchar[0], vchar[1], vchar[2]});
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", {vbuf[0], vbuf[1], vbuf[2]});
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", {vstring[0], vstring[1], vstring[2]});
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }

        // initializer lists with explicit types
        UNIT_ASSERT_EQUAL(JoinSeq(" ", std::initializer_list<const char*>{v[0], v[1], v[2]}), result);
        UNIT_ASSERT_EQUAL(JoinSeq(" ", std::initializer_list<const char*>{vchar[0], vchar[1], vchar[2]}), result);
        UNIT_ASSERT_EQUAL(JoinSeq(" ", std::initializer_list<TStringBuf>{vbuf[0], vbuf[1], vbuf[2]}), result);
        UNIT_ASSERT_EQUAL(JoinSeq(" ", std::initializer_list<TString>{vstring[0], vstring[1], vstring[2]}), result);
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", std::initializer_list<const char*>{v[0], v[1], v[2]});
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", std::initializer_list<const char*>{vchar[0], vchar[1], vchar[2]});
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", std::initializer_list<TStringBuf>{vbuf[0], vbuf[1], vbuf[2]});
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }
        {
            TStringStream stream;
            stream << MakeRangeJoiner(" ", std::initializer_list<TString>{vstring[0], vstring[1], vstring[2]});
            UNIT_ASSERT_EQUAL(stream.Str(), result);
        }

        // c-style array
        UNIT_ASSERT_VALUES_EQUAL(JoinSeq(" ", v), result);
    }

    Y_UNIT_TEST(CustomToString) {
        TCustomData d1{{1, 2, 3, 4, 5}};
        TCustomData d2{{0, -1, -2}};
        UNIT_ASSERT_EQUAL(Join(" ", d1, d2), "1__2__3__4__5 0__-1__-2");
    }

    Y_UNIT_TEST(JoinChars) {
        // Note that char delimeter is printed as single char string,
        // but joined char values are printed as their numeric codes! O_o
        UNIT_ASSERT_EQUAL(Join('a', 'a', 'a'), "97a97");
        UNIT_ASSERT_EQUAL(Join("a", "a", "a"), "aaa");
    }
} // Y_UNIT_TEST_SUITE(JoinStringTest)
