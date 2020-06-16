#include <library/cpp/dbg_output/dump.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/str.h>
#include <util/string/builder.h>
#include <util/string/escape.h>
#include <util/generic/map.h>

namespace {
    struct TX {
        inline TX() {
            N = this;
        }

        TX* N;
    };
}

template <>
struct TDumper<TX> {
    template <class S>
    static inline void Dump(S& s, const TX& x) {
        s << DumpRaw("x") << x.N;
    }
};

namespace TMyNS {
    struct TMyStruct {
        int A, B;
    };
}
DEFINE_DUMPER(TMyNS::TMyStruct, A, B)

Y_UNIT_TEST_SUITE(TContainerPrintersTest) {
    Y_UNIT_TEST(TestVectorInt) {
        TStringStream out;
        out << DbgDump(TVector<int>({1, 2, 3, 4, 5}));
        UNIT_ASSERT_STRINGS_EQUAL(out.Str(), "[1, 2, 3, 4, 5]");
    }

    Y_UNIT_TEST(TestMapCharToCharArray) {
        TStringStream out;

        TMap<char, const char*> m;

        m['a'] = "SMALL LETTER A";
        m['b'] = nullptr;

        out << DbgDump(m);

        UNIT_ASSERT_STRINGS_EQUAL(out.Str(), "{'a' -> \"SMALL LETTER A\", 'b' -> (empty)}");
    }

    Y_UNIT_TEST(TestVectorOfVectors) {
        TStringStream out;
        TVector<TVector<wchar16>> vec(2);
        vec[0].push_back(0);
        vec[1] = {wchar16('a')};
        out << DbgDump(vec);
        UNIT_ASSERT_STRINGS_EQUAL(out.Str(), "[[w'\\0'], [w'a']]");
    }

    Y_UNIT_TEST(TestInfinite) {
        UNIT_ASSERT(!!(TStringBuilder() << DbgDumpDeep(TX())));
    }

    Y_UNIT_TEST(TestLabeledDump) {
        TStringStream out;
        int a = 1, b = 2;
        out << LabeledDump(a, b, 1 + 2);
        UNIT_ASSERT_STRINGS_EQUAL(out.Str(), "{\"a\": 1, \"b\": 2, \"1 + 2\": 3}");
    }

    Y_UNIT_TEST(TestStructDumper) {
        TStringStream out;
        out << DbgDump(TMyNS::TMyStruct{3, 4});
        UNIT_ASSERT_STRINGS_EQUAL(out.Str(), "{\"A\": 3, \"B\": 4}");
    }

    Y_UNIT_TEST(TestColors) {
        using TComplex = TMap<TString, TMap<int, char>>;
        TComplex test;
        test["a"][1] = '7';
        test["b"][2] = '6';
        TStringStream out;
        out << DbgDump<TComplex, NDbgDump::NColorScheme::TEyebleed</* Enforce = */ true>>(test);
        UNIT_ASSERT_STRINGS_EQUAL(
            EscapeC(out.Str()),
            "\\x1B[1;32m{\\x1B[1;31m\\x1B[42m\\x1B[1;31m\\x1B[1;33m\\\"a\\\"\\x1B[22;39m\\x1B[22;39m"
            "\\x1B[49m\\x1B[1;32m -> \\x1B[44m\\x1B[1;31m\\x1B[1;32m{\\x1B[1;31m\\x1B[1;31m1"
            "\\x1B[22;39m\\x1B[1;32m -> \\x1B[1;31m'7'\\x1B[22;39m\\x1B[1;32m}"
            "\\x1B[22;39m\\x1B[22;39m\\x1B[49m\\x1B[1;32m, \\x1B[1;31m\\x1B[42m\\x1B[1;31m\\x1B[1;33m"
            "\\\"b\\\"\\x1B[22;39m\\x1B[22;39m\\x1B[49m\\x1B[1;32m -> "
            "\\x1B[44m\\x1B[1;31m\\x1B[1;32m{\\x1B[1;31m\\x1B[1;31m2\\x1B[22;39m\\x1B[1;32m -> "
            "\\x1B[1;31m'6'\\x1B[22;39m\\x1B[1;32m}\\x1B[22;39m\\x1B[22;39m\\x1B[49m\\x1B[1;32m}\\x1B[22;39m");
    }

    Y_UNIT_TEST(SmallIntOrChar) {
        char c = 'e';
        i8 i = -100;
        ui8 u = 10;
        UNIT_ASSERT_VALUES_EQUAL(TStringBuilder() << DbgDump(c), "'e'");
        UNIT_ASSERT_VALUES_EQUAL(TStringBuilder() << DbgDump(i), "-100");
        UNIT_ASSERT_VALUES_EQUAL(TStringBuilder() << DbgDump(u), "10");
    }
}
