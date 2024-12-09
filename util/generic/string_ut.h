#pragma once

#include "string.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/string/reverse.h>

template <typename CharT, size_t N>
struct TCharBuffer {
    CharT Data[N];
    //! copies characters from string to the internal buffer without any conversion
    //! @param s    a string that must contain only characters less than 0x7F
    explicit TCharBuffer(const char* s) {
        // copy all symbols including null terminated symbol
        for (size_t i = 0; i < N; ++i) {
            Data[i] = s[i];
        }
    }
    const CharT* GetData() const {
        return Data;
    }
};

template <>
struct TCharBuffer<char, 0> {
    const char* Data;
    //! stores pointer to string
    explicit TCharBuffer(const char* s)
        : Data(s)
    {
    }
    const char* GetData() const {
        return Data;
    }
};

#define DECLARE_AND_RETURN_BUFFER(s)             \
    static TCharBuffer<CharT, sizeof(s)> buf(s); \
    return buf.GetData();

//! @attention this class can process characters less than 0x7F only (the low half of ASCII table)
template <typename CharT>
struct TTestData {
    // words
    const CharT* str1() {
        DECLARE_AND_RETURN_BUFFER("str1");
    }
    const CharT* str2() {
        DECLARE_AND_RETURN_BUFFER("str2");
    }
    const CharT* str__________________________________________________1() {
        DECLARE_AND_RETURN_BUFFER("str                                                  1");
    }
    const CharT* str__________________________________________________2() {
        DECLARE_AND_RETURN_BUFFER("str                                                  2");
    }
    const CharT* one() {
        DECLARE_AND_RETURN_BUFFER("one");
    }
    const CharT* two() {
        DECLARE_AND_RETURN_BUFFER("two");
    }
    const CharT* three() {
        DECLARE_AND_RETURN_BUFFER("three");
    }
    const CharT* thrii() {
        DECLARE_AND_RETURN_BUFFER("thrii");
    }
    const CharT* four() {
        DECLARE_AND_RETURN_BUFFER("four");
    }
    const CharT* enotw_() {
        DECLARE_AND_RETURN_BUFFER("enotw ");
    }
    const CharT* foo() {
        DECLARE_AND_RETURN_BUFFER("foo");
    }
    const CharT* abcdef() {
        DECLARE_AND_RETURN_BUFFER("abcdef");
    }
    const CharT* abcdefg() {
        DECLARE_AND_RETURN_BUFFER("abcdefg");
    }
    const CharT* aba() {
        DECLARE_AND_RETURN_BUFFER("aba");
    }
    const CharT* hr() {
        DECLARE_AND_RETURN_BUFFER("hr");
    }
    const CharT* hrt() {
        DECLARE_AND_RETURN_BUFFER("hrt");
    }
    const CharT* thr() {
        DECLARE_AND_RETURN_BUFFER("thr");
    }
    const CharT* tw() {
        DECLARE_AND_RETURN_BUFFER("tw");
    }
    const CharT* ow() {
        DECLARE_AND_RETURN_BUFFER("ow");
    }
    const CharT* opq() {
        DECLARE_AND_RETURN_BUFFER("opq");
    }
    const CharT* xyz() {
        DECLARE_AND_RETURN_BUFFER("xyz");
    }
    const CharT* abc() {
        DECLARE_AND_RETURN_BUFFER("abc");
    }
    const CharT* abcd() {
        DECLARE_AND_RETURN_BUFFER("abcd");
    }
    const CharT* abcde() {
        DECLARE_AND_RETURN_BUFFER("abcde");
    }
    const CharT* abcc() {
        DECLARE_AND_RETURN_BUFFER("abcc");
    }
    const CharT* abce() {
        DECLARE_AND_RETURN_BUFFER("abce");
    }
    const CharT* qwe() {
        DECLARE_AND_RETURN_BUFFER("qwe");
    }
    const CharT* cd() {
        DECLARE_AND_RETURN_BUFFER("cd");
    }
    const CharT* cde() {
        DECLARE_AND_RETURN_BUFFER("cde");
    }
    const CharT* cdef() {
        DECLARE_AND_RETURN_BUFFER("cdef");
    }
    const CharT* cdefgh() {
        DECLARE_AND_RETURN_BUFFER("cdefgh");
    }
    const CharT* ehortw_() {
        DECLARE_AND_RETURN_BUFFER("ehortw ");
    }
    const CharT* fg() {
        DECLARE_AND_RETURN_BUFFER("fg");
    }
    const CharT* abcdefgh() {
        DECLARE_AND_RETURN_BUFFER("abcdefgh");
    }

    // phrases
    const CharT* Hello_World() {
        DECLARE_AND_RETURN_BUFFER("Hello World");
    }
    const CharT* This_is_test_string_for_string_calls() {
        DECLARE_AND_RETURN_BUFFER("This is test string for string calls");
    }
    const CharT* This_is_teis_test_string_st_string_for_string_calls() {
        DECLARE_AND_RETURN_BUFFER("This is teis test string st string for string calls");
    }
    const CharT* This_is_test_stis_test_string_for_stringring_for_string_calls() {
        DECLARE_AND_RETURN_BUFFER("This is test stis test string for stringring for string calls");
    }
    const CharT* allsThis_is_test_string_for_string_calls() {
        DECLARE_AND_RETURN_BUFFER("allsThis is test string for string calls");
    }
    const CharT* ng_for_string_callsThis_is_test_string_for_string_calls() {
        DECLARE_AND_RETURN_BUFFER("ng for string callsThis is test string for string calls");
    }
    const CharT* one_two_three_one_two_three() {
        DECLARE_AND_RETURN_BUFFER("one two three one two three");
    }
    const CharT* test_string_for_assign() {
        DECLARE_AND_RETURN_BUFFER("test string for assign");
    }
    const CharT* other_test_string() {
        DECLARE_AND_RETURN_BUFFER("other test string");
    }
    const CharT* This_This_is_tefor_string_calls() {
        DECLARE_AND_RETURN_BUFFER("This This is tefor string calls");
    }
    const CharT* This_This_is_test_string_for_string_calls() {
        DECLARE_AND_RETURN_BUFFER("This This is test string for string calls");
    }

    const CharT* _0123456x() {
        DECLARE_AND_RETURN_BUFFER("0123456x");
    }
    const CharT* _0123456xy() {
        DECLARE_AND_RETURN_BUFFER("0123456xy");
    }
    const CharT* _0123456xyz() {
        DECLARE_AND_RETURN_BUFFER("0123456xyz");
    }
    const CharT* _0123456xyzZ() {
        DECLARE_AND_RETURN_BUFFER("0123456xyzZ");
    }
    const CharT* _0123456xyzZ0() {
        DECLARE_AND_RETURN_BUFFER("0123456xyzZ0");
    }
    const CharT* abc0123456xyz() {
        DECLARE_AND_RETURN_BUFFER("abc0123456xyz");
    }
    const CharT* BCabc0123456xyz() {
        DECLARE_AND_RETURN_BUFFER("BCabc0123456xyz");
    }
    const CharT* qweBCabc0123456xyz() {
        DECLARE_AND_RETURN_BUFFER("qweBCabc0123456xyz");
    }
    const CharT* _1qweBCabc0123456xyz() {
        DECLARE_AND_RETURN_BUFFER("1qweBCabc0123456xyz");
    }
    const CharT* _01abc23456() {
        DECLARE_AND_RETURN_BUFFER("01abc23456");
    }
    const CharT* _01ABCabc23456() {
        DECLARE_AND_RETURN_BUFFER("01ABCabc23456");
    }
    const CharT* ABC() {
        DECLARE_AND_RETURN_BUFFER("ABC");
    }
    const CharT* ABCD() {
        DECLARE_AND_RETURN_BUFFER("ABCD");
    }
    const CharT* QWE() {
        DECLARE_AND_RETURN_BUFFER("QWE");
    }
    const CharT* XYZ() {
        DECLARE_AND_RETURN_BUFFER("XYZ");
    }
    const CharT* W01ABCabc23456() {
        DECLARE_AND_RETURN_BUFFER("W01ABCabc23456");
    }
    const CharT* abcd456() {
        DECLARE_AND_RETURN_BUFFER("abcd456");
    }
    const CharT* abcdABCD() {
        DECLARE_AND_RETURN_BUFFER("abcdABCD");
    }
    const CharT* abcdABC123() {
        DECLARE_AND_RETURN_BUFFER("abcdABC123");
    }
    const CharT* z123z123() {
        DECLARE_AND_RETURN_BUFFER("z123z123");
    }
    const CharT* ASDF1234QWER() {
        DECLARE_AND_RETURN_BUFFER("ASDF1234QWER");
    }
    const CharT* asdf1234qwer() {
        DECLARE_AND_RETURN_BUFFER("asdf1234qwer");
    }
    const CharT* asDF1234qWEr() {
        DECLARE_AND_RETURN_BUFFER("asDF1234qWEr");
    }
    const CharT* AsDF1234qWEr() {
        DECLARE_AND_RETURN_BUFFER("AsDF1234qWEr");
    }
    const CharT* Asdf1234qwer() {
        DECLARE_AND_RETURN_BUFFER("Asdf1234qwer");
    }
    const CharT* Asdf1234qwerWWWW() {
        DECLARE_AND_RETURN_BUFFER("Asdf1234qwerWWWW");
    }
    const CharT* Asdf() {
        DECLARE_AND_RETURN_BUFFER("Asdf");
    }
    const CharT* orig() {
        DECLARE_AND_RETURN_BUFFER("orig");
    }
    const CharT* fdfdsfds() {
        DECLARE_AND_RETURN_BUFFER("fdfdsfds");
    }

    // numbers
    const CharT* _0() {
        DECLARE_AND_RETURN_BUFFER("0");
    }
    const CharT* _00() {
        DECLARE_AND_RETURN_BUFFER("00");
    }
    const CharT* _0000000000() {
        DECLARE_AND_RETURN_BUFFER("0000000000");
    }
    const CharT* _00000() {
        DECLARE_AND_RETURN_BUFFER("00000");
    }
    const CharT* _0123() {
        DECLARE_AND_RETURN_BUFFER("0123");
    }
    const CharT* _01230123() {
        DECLARE_AND_RETURN_BUFFER("01230123");
    }
    const CharT* _01234() {
        DECLARE_AND_RETURN_BUFFER("01234");
    }
    const CharT* _0123401234() {
        DECLARE_AND_RETURN_BUFFER("0123401234");
    }
    const CharT* _012345() {
        DECLARE_AND_RETURN_BUFFER("012345");
    }
    const CharT* _0123456() {
        DECLARE_AND_RETURN_BUFFER("0123456");
    }
    const CharT* _1() {
        DECLARE_AND_RETURN_BUFFER("1");
    }
    const CharT* _11() {
        DECLARE_AND_RETURN_BUFFER("11");
    }
    const CharT* _1100000() {
        DECLARE_AND_RETURN_BUFFER("1100000");
    }
    const CharT* _110000034() {
        DECLARE_AND_RETURN_BUFFER("110000034");
    }
    const CharT* _12() {
        DECLARE_AND_RETURN_BUFFER("12");
    }
    const CharT* _123() {
        DECLARE_AND_RETURN_BUFFER("123");
    }
    const CharT* _1233321() {
        DECLARE_AND_RETURN_BUFFER("1233321");
    }
    const CharT* _1221() {
        DECLARE_AND_RETURN_BUFFER("1221");
    }
    const CharT* _1234123456() {
        DECLARE_AND_RETURN_BUFFER("1234123456");
    }
    const CharT* _12334444321() {
        DECLARE_AND_RETURN_BUFFER("12334444321");
    }
    const CharT* _123344544321() {
        DECLARE_AND_RETURN_BUFFER("123344544321");
    }
    const CharT* _1234567890123456789012345678901234567890() {
        DECLARE_AND_RETURN_BUFFER("1234567890123456789012345678901234567890");
    }
    const CharT* _1234() {
        DECLARE_AND_RETURN_BUFFER("1234");
    }
    const CharT* _12345() {
        DECLARE_AND_RETURN_BUFFER("12345");
    }
    const CharT* _123456() {
        DECLARE_AND_RETURN_BUFFER("123456");
    }
    const CharT* _1234567() {
        DECLARE_AND_RETURN_BUFFER("1234567");
    }
    const CharT* _1234561234() {
        DECLARE_AND_RETURN_BUFFER("1234561234");
    }
    const CharT* _12356() {
        DECLARE_AND_RETURN_BUFFER("12356");
    }
    const CharT* _1345656() {
        DECLARE_AND_RETURN_BUFFER("1345656");
    }
    const CharT* _15656() {
        DECLARE_AND_RETURN_BUFFER("15656");
    }
    const CharT* _17856() {
        DECLARE_AND_RETURN_BUFFER("17856");
    }
    const CharT* _1783456() {
        DECLARE_AND_RETURN_BUFFER("1783456");
    }
    const CharT* _2() {
        DECLARE_AND_RETURN_BUFFER("2");
    }
    const CharT* _2123456() {
        DECLARE_AND_RETURN_BUFFER("2123456");
    }
    const CharT* _23() {
        DECLARE_AND_RETURN_BUFFER("23");
    }
    const CharT* _2345() {
        DECLARE_AND_RETURN_BUFFER("2345");
    }
    const CharT* _3() {
        DECLARE_AND_RETURN_BUFFER("3");
    }
    const CharT* _345() {
        DECLARE_AND_RETURN_BUFFER("345");
    }
    const CharT* _3456() {
        DECLARE_AND_RETURN_BUFFER("3456");
    }
    const CharT* _333333() {
        DECLARE_AND_RETURN_BUFFER("333333");
    }
    const CharT* _389() {
        DECLARE_AND_RETURN_BUFFER("389");
    }
    const CharT* _4294967295() {
        DECLARE_AND_RETURN_BUFFER("4294967295");
    }
    const CharT* _4444() {
        DECLARE_AND_RETURN_BUFFER("4444");
    }
    const CharT* _5() {
        DECLARE_AND_RETURN_BUFFER("5");
    }
    const CharT* _6() {
        DECLARE_AND_RETURN_BUFFER("6");
    }
    const CharT* _6543210() {
        DECLARE_AND_RETURN_BUFFER("6543210");
    }
    const CharT* _7() {
        DECLARE_AND_RETURN_BUFFER("7");
    }
    const CharT* _78() {
        DECLARE_AND_RETURN_BUFFER("78");
    }
    const CharT* _2004_01_01() {
        DECLARE_AND_RETURN_BUFFER("2004-01-01");
    }
    const CharT* _1234562004_01_01() {
        DECLARE_AND_RETURN_BUFFER("1234562004-01-01");
    }
    const CharT* _0123456_12345() {
        DECLARE_AND_RETURN_BUFFER("0123456_12345");
    }

    // letters
    const CharT* a() {
        DECLARE_AND_RETURN_BUFFER("a");
    }
    const CharT* b() {
        DECLARE_AND_RETURN_BUFFER("b");
    }
    const CharT* c() {
        DECLARE_AND_RETURN_BUFFER("c");
    }
    const CharT* d() {
        DECLARE_AND_RETURN_BUFFER("d");
    }
    const CharT* e() {
        DECLARE_AND_RETURN_BUFFER("e");
    }
    const CharT* f() {
        DECLARE_AND_RETURN_BUFFER("f");
    }
    const CharT* h() {
        DECLARE_AND_RETURN_BUFFER("h");
    }
    const CharT* o() {
        DECLARE_AND_RETURN_BUFFER("o");
    }
    const CharT* p() {
        DECLARE_AND_RETURN_BUFFER("p");
    }
    const CharT* q() {
        DECLARE_AND_RETURN_BUFFER("q");
    }
    const CharT* r() {
        DECLARE_AND_RETURN_BUFFER("r");
    }
    const CharT* s() {
        DECLARE_AND_RETURN_BUFFER("s");
    }
    const CharT* t() {
        DECLARE_AND_RETURN_BUFFER("t");
    }
    const CharT* w() {
        DECLARE_AND_RETURN_BUFFER("w");
    }
    const CharT* x() {
        DECLARE_AND_RETURN_BUFFER("x");
    }
    const CharT* y() {
        DECLARE_AND_RETURN_BUFFER("y");
    }
    const CharT* z() {
        DECLARE_AND_RETURN_BUFFER("z");
    }
    const CharT* H() {
        DECLARE_AND_RETURN_BUFFER("H");
    }
    const CharT* I() {
        DECLARE_AND_RETURN_BUFFER("I");
    }
    const CharT* W() {
        DECLARE_AND_RETURN_BUFFER("W");
    }

    const CharT* Space() {
        DECLARE_AND_RETURN_BUFFER(" ");
    }
    const CharT* Empty() {
        DECLARE_AND_RETURN_BUFFER("");
    }

    size_t HashOf_0123456() {
        return 0;
    }
};

template <>
size_t TTestData<char>::HashOf_0123456() {
    return 1229863857ul;
}

template <>
size_t TTestData<wchar16>::HashOf_0123456() {
    return 2775195331ul;
}

template <class TStringType, typename TTestData>
class TStringTestImpl {
protected:
    using char_type = typename TStringType::char_type;
    using traits_type = typename TStringType::traits_type;

    TTestData Data;

public:
    void TestMaxSize() {
        const size_t badMaxVal = TStringType{}.max_size() + 1;

        TStringType s;
        UNIT_CHECK_GENERATED_EXCEPTION(s.reserve(badMaxVal), std::length_error);
    }

    void TestConstructors() {
        TStringType s0;
        UNIT_ASSERT(s0.size() == 0);

        TStringType s;
        TStringType s1(*Data._0());
        TStringType s2(Data._0());
        UNIT_ASSERT(s1 == s2);

        TStringType fromChar(char_type('a'));
        UNIT_ASSERT_VALUES_EQUAL(fromChar.size(), 1u);
        UNIT_ASSERT_VALUES_EQUAL(fromChar[0], char_type('a'));

#ifndef TSTRING_IS_STD_STRING
        TStringType s3 = TStringType::Uninitialized(10);
        UNIT_ASSERT(s3.size() == 10);
#endif

        TStringType s4(Data._0123456(), 1, 3);
        UNIT_ASSERT(s4 == Data._123());

        TStringType s5(5, *Data._0());
        UNIT_ASSERT(s5 == Data._00000());

        TStringType s6(Data._0123456());
        UNIT_ASSERT(s6 == Data._0123456());
        TStringType s7(s6);
        UNIT_ASSERT(s7 == s6);
#ifndef TSTRING_IS_STD_STRING
        UNIT_ASSERT(s7.c_str() == s6.c_str());
#endif

        TStringType s8(s7, 1, 3);
        UNIT_ASSERT(s8 == Data._123());

        TStringType s9(*Data._1());
        UNIT_ASSERT(s9 == Data._1());

        TStringType s10(Reserve(100));
        UNIT_ASSERT(s10.empty());
        UNIT_ASSERT(s10.capacity() >= 100);
    }

    void TestReplace() {
        TStringType s(Data._0123456());
        UNIT_ASSERT(s.copy() == Data._0123456());

        // append family
        s.append(Data.x());
        UNIT_ASSERT(s == Data._0123456x());

#ifdef TSTRING_IS_STD_STRING
        s.append(Data.xyz() + 1, 1);
#else
        s.append(Data.xyz(), 1, 1);
#endif
        UNIT_ASSERT(s == Data._0123456xy());

        s.append(TStringType(Data.z()));
        UNIT_ASSERT(s == Data._0123456xyz());

        s.append(TStringType(Data.XYZ()), 2, 1);
        UNIT_ASSERT(s == Data._0123456xyzZ());

        s.append(*Data._0());
        UNIT_ASSERT(s == Data._0123456xyzZ0());

        // prepend family
        s = Data._0123456xyz();
        s.prepend(TStringType(Data.abc()));
        UNIT_ASSERT(s == Data.abc0123456xyz());

        s.prepend(TStringType(Data.ABC()), 1, 2);
        UNIT_ASSERT(s == Data.BCabc0123456xyz());

        s.prepend(Data.qwe());
        UNIT_ASSERT(s == Data.qweBCabc0123456xyz());

        s.prepend(*Data._1());
        UNIT_ASSERT(s == Data._1qweBCabc0123456xyz());

        // substr
        s = Data.abc0123456xyz();
        s = s.substr(3, 7);
        UNIT_ASSERT(s == Data._0123456());

        // insert family
        s.insert(2, Data.abc());
        UNIT_ASSERT(s == Data._01abc23456());

        s.insert(2, TStringType(Data.ABC()));
        UNIT_ASSERT(s == Data._01ABCabc23456());

        s.insert(0, TStringType(Data.QWE()), 1, 1);
        UNIT_ASSERT(s == Data.W01ABCabc23456());

        // replace family
        s = Data._01abc23456();
        s.replace(0, 7, Data.abcd());
        UNIT_ASSERT(s == Data.abcd456());

        s.replace(4, 3, TStringType(Data.ABCD()));
        UNIT_ASSERT(s == Data.abcdABCD());

        s.replace(7, 10, TStringType(Data._01234()), 1, 3);
        UNIT_ASSERT(s == Data.abcdABC123());
        UNIT_ASSERT(Data.abcdABC123() == s);

        // remove, erase
        s.remove(4);
        UNIT_ASSERT(s == Data.abcd());
        s.erase(3);
        UNIT_ASSERT(s == Data.abc());

        // Read access
        s = Data._012345();
        UNIT_ASSERT(s.at(1) == *Data._1());
        UNIT_ASSERT(s[1] == *Data._1());
        UNIT_ASSERT(s.at(s.size()) == 0);
        UNIT_ASSERT(s[s.size()] == 0);
    }

#ifndef TSTRING_IS_STD_STRING
    void TestRefCount() {
        using TStr = TStringType;

        struct TestStroka: public TStr {
            using TStr::TStr;
            // un-protect
            using TStr::RefCount;
        };

        TestStroka s1(Data.orig());
        UNIT_ASSERT_EQUAL(s1.RefCount() == 1, true);
        TestStroka s2(s1);
        UNIT_ASSERT_EQUAL(s1.RefCount() == 2, true);
        UNIT_ASSERT_EQUAL(s2.RefCount() == 2, true);
        UNIT_ASSERT_EQUAL(s1.c_str() == s2.c_str(), true); // the same pointer
        char_type* beg = s2.begin();
        UNIT_ASSERT_EQUAL(s1 == beg, true);
        UNIT_ASSERT_EQUAL(s1.RefCount() == 1, true);
        UNIT_ASSERT_EQUAL(s2.RefCount() == 1, true);
        UNIT_ASSERT_EQUAL(s1.c_str() == s2.c_str(), false);
    }
#endif

    // Find family

    void TestFind() {
        const TStringType s(Data._0123456_12345());
        const TStringType s2(Data._0123());

        UNIT_ASSERT(s.find(Data._345()) == 3);
        UNIT_ASSERT(s.find(Data._345(), 5) == 10);

        UNIT_ASSERT(s.find(Data._345(), 20) == TStringType::npos);
        UNIT_ASSERT(s.find(*Data._3()) == 3);
        UNIT_ASSERT(s.find(TStringType(Data._345())) == 3);
        UNIT_ASSERT(s.find(TStringType(Data._345()), 2) == 3);

        UNIT_ASSERT(s.find_first_of(TStringType(Data._389())) == 3);
        UNIT_ASSERT(s.find_first_of(Data._389()) == 3);
        UNIT_ASSERT(s.find_first_of(Data._389(), s.size()) == TStringType::npos);
        UNIT_ASSERT(s.find_first_not_of(Data._123()) == 0);
        UNIT_ASSERT(s.find_first_of('6') == 6);
        UNIT_ASSERT(s.find_first_of('1', 2) == 8);
        UNIT_ASSERT(s.find_first_not_of('0') == 1);
        UNIT_ASSERT(s.find_first_not_of('1', 1) == 2);

        const TStringType rs = Data._0123401234();
        UNIT_ASSERT(rs.rfind(*Data._3()) == 8);

        const TStringType empty;
        UNIT_ASSERT(empty.find(empty) == 0);
        UNIT_ASSERT(s.find(empty, 0) == 0);
        UNIT_ASSERT(s.find(empty, 1) == 1);
        UNIT_ASSERT(s.find(empty, s.length()) == s.length());
        UNIT_ASSERT(s.find(empty, s.length() + 1) == TStringType::npos);

        UNIT_ASSERT(s.rfind(empty) == s.length());
        UNIT_ASSERT(empty.rfind(empty) == 0);
        UNIT_ASSERT(empty.rfind(s) == TStringType::npos);

        UNIT_ASSERT(s2.rfind(s) == TStringType::npos);
        UNIT_ASSERT(s.rfind(s2) == 0);
        UNIT_ASSERT(s.rfind(TStringType(Data._345())) == 10);
        UNIT_ASSERT(s.rfind(TStringType(Data._345()), 13) == 10);
        UNIT_ASSERT(s.rfind(TStringType(Data._345()), 10) == 10);
        UNIT_ASSERT(s.rfind(TStringType(Data._345()), 9) == 3);
        UNIT_ASSERT(s.rfind(TStringType(Data._345()), 6) == 3);
        UNIT_ASSERT(s.rfind(TStringType(Data._345()), 3) == 3);
        UNIT_ASSERT(s.rfind(TStringType(Data._345()), 2) == TStringType::npos);
    }

    void TestContains() {
        const TStringType s(Data._0123456_12345());
        const TStringType s2(Data._0123());

        UNIT_ASSERT(s.Contains(Data._345()));
        UNIT_ASSERT(!s2.Contains(Data._345()));

        UNIT_ASSERT(s.Contains('1'));
        UNIT_ASSERT(!s.Contains('*'));

        TStringType empty;
        UNIT_ASSERT(s.Contains(empty));
        UNIT_ASSERT(!empty.Contains(s));
        UNIT_ASSERT(empty.Contains(empty));
        UNIT_ASSERT(s.Contains(empty, s.length()));
    }

    // Operators

    void TestOperators() {
        TStringType s(Data._0123456());

        // operator +=
        s += TStringType(Data.x());
        UNIT_ASSERT(s == Data._0123456x());

        s += Data.y();
        UNIT_ASSERT(s == Data._0123456xy());

        s += *Data.z();
        UNIT_ASSERT(s == Data._0123456xyz());

        // operator +
        s = Data._0123456();
        s = s + TStringType(Data.x());
        UNIT_ASSERT(s == Data._0123456x());

        s = s + Data.y();
        UNIT_ASSERT(s == Data._0123456xy());

        s = s + *Data.z();
        UNIT_ASSERT(s == Data._0123456xyz());

        // operator !=
        s = Data._012345();
        UNIT_ASSERT(s != TStringType(Data.xyz()));
        UNIT_ASSERT(s != Data.xyz());
        UNIT_ASSERT(Data.xyz() != s);

        // operator <
        UNIT_ASSERT_EQUAL(s < TStringType(Data.xyz()), true);
        UNIT_ASSERT_EQUAL(s < Data.xyz(), true);
        UNIT_ASSERT_EQUAL(Data.xyz() < s, false);

        // operator <=
        UNIT_ASSERT_EQUAL(s <= TStringType(Data.xyz()), true);
        UNIT_ASSERT_EQUAL(s <= Data.xyz(), true);
        UNIT_ASSERT_EQUAL(Data.xyz() <= s, false);

        // operator >
        UNIT_ASSERT_EQUAL(s > TStringType(Data.xyz()), false);
        UNIT_ASSERT_EQUAL(s > Data.xyz(), false);
        UNIT_ASSERT_EQUAL(Data.xyz() > s, true);

        // operator >=
        UNIT_ASSERT_EQUAL(s >= TStringType(Data.xyz()), false);
        UNIT_ASSERT_EQUAL(s >= Data.xyz(), false);
        UNIT_ASSERT_EQUAL(Data.xyz() >= s, true);
    }

    void TestOperatorsCI() {
        TStringType s(Data.ABCD());
        UNIT_ASSERT(s > Data.abc0123456xyz());
        UNIT_ASSERT(s == Data.abcd());

        using TCIStringBuf = TBasicStringBuf<char_type, traits_type>;

        UNIT_ASSERT(s > TCIStringBuf(Data.abc0123456xyz()));
        UNIT_ASSERT(TCIStringBuf(Data.abc0123456xyz()) < s);
        UNIT_ASSERT(s == TCIStringBuf(Data.abcd()));
    }

    void TestMulOperators() {
        {
            TStringType s(Data._0());
            s *= 10;
            UNIT_ASSERT_EQUAL(s, TStringType(Data._0000000000()));
        }
        {
            TStringType s = TStringType(Data._0()) * 2;
            UNIT_ASSERT_EQUAL(s, TStringType(Data._00()));
        }
    }

    // Test any other functions

    void TestFuncs() {
        TStringType s(Data._0123456());
        UNIT_ASSERT(s.c_str() == s.data());

        // length()
        UNIT_ASSERT(s.length() == s.size());
        UNIT_ASSERT(s.length() == traits_type::length(s.data()));

        // is_null()
        TStringType s1(Data.Empty());
        UNIT_ASSERT(s1.is_null() == true);
        UNIT_ASSERT(s1.is_null() == s1.empty());
        UNIT_ASSERT(s1.is_null() == !s1);

        TStringType s2(s);
        UNIT_ASSERT(s2 == s);

        // reverse()
        ReverseInPlace(s2);
        UNIT_ASSERT(s2 == Data._6543210());

        // to_upper()
        s2 = Data.asdf1234qwer();
        s2.to_upper();
        UNIT_ASSERT(s2 == Data.ASDF1234QWER());

        // to_lower()
        s2.to_lower();
        UNIT_ASSERT(s2 == Data.asdf1234qwer());

        // to_title()
        s2 = Data.asDF1234qWEr();
        s2.to_title();
        UNIT_ASSERT(s2 == Data.Asdf1234qwer());

        s2 = Data.AsDF1234qWEr();
        s2.to_title();
        UNIT_ASSERT(s2 == Data.Asdf1234qwer());

        // Friend functions
        s2 = Data.asdf1234qwer();
        TStringType s3 = to_upper(s2);
        UNIT_ASSERT(s3 == Data.ASDF1234QWER());
        s3 = to_lower(s2);
        UNIT_ASSERT(s3 == Data.asdf1234qwer());
        s3 = to_title(s2);
        UNIT_ASSERT(s3 == Data.Asdf1234qwer());
        s2 = s3;

        // resize family
        s2.resize(s2.size()); // without length change
        UNIT_ASSERT(s2 == Data.Asdf1234qwer());

        s2.resize(s2.size() + 4, *Data.W());
        UNIT_ASSERT(s2 == Data.Asdf1234qwerWWWW());

        s2.resize(4);
        UNIT_ASSERT(s2 == Data.Asdf());

        // assign family
        s2 = Data.asdf1234qwer();
        s2.assign(s, 1, 3);
        UNIT_ASSERT(s2 == Data._123());

        s2.assign(Data._0123456(), 4);
        UNIT_ASSERT(s2 == Data._0123());

        s2.assign('1');
        UNIT_ASSERT(s2 == Data._1());

        s2.assign(Data._0123456());
        UNIT_ASSERT(s2 == Data._0123456());

        // hash()
        TStringType sS = s2; // type 'TStringType' is used as is

        ComputeHash(sS); /*size_t hash_val = sS.hash();

        try {
            // UNIT_ASSERT(hash_val == Data.HashOf_0123456());
        } catch (...) {
            Cerr << hash_val << Endl;
            throw;
        }*/

        s2.assign(Data._0123456(), 2, 2);
        UNIT_ASSERT(s2 == Data._23());

        // s2.reserve();

        TStringType s5(Data.abcde());
        s5.clear();
        UNIT_ASSERT(s5 == Data.Empty());
    }

    void TestUtils() {
        TStringType s;
        s = Data._01230123();
        TStringType from = Data._0();
        TStringType to = Data.z();

        SubstGlobal(s, from, to);
        UNIT_ASSERT(s == Data.z123z123());
    }

    void TestEmpty() {
        TStringType s;
        s = Data._2();
        s = TStringType(Data.fdfdsfds(), (size_t)0, (size_t)0);
        UNIT_ASSERT(s.empty());
    }

    void TestJoin() {
        UNIT_ASSERT_EQUAL(TStringType::Join(Data._12(), Data._3456()), Data._123456());
        UNIT_ASSERT_EQUAL(TStringType::Join(Data._12(), TStringType(Data._3456())), Data._123456());
        UNIT_ASSERT_EQUAL(TStringType::Join(TStringType(Data._12()), Data._3456()), Data._123456());
        UNIT_ASSERT_EQUAL(TStringType::Join(Data._12(), Data._345(), Data._6()), Data._123456());
        UNIT_ASSERT_EQUAL(TStringType::Join(Data._12(), TStringType(Data._345()), Data._6()), Data._123456());
        UNIT_ASSERT_EQUAL(TStringType::Join(TStringType(Data._12()), TStringType(Data._345()), Data._6()), Data._123456());
        UNIT_ASSERT_EQUAL(TStringType::Join(TStringType(Data.a()), Data.b(), TStringType(Data.cd()), TStringType(Data.e()), Data.fg(), TStringType(Data.h())), Data.abcdefgh());
        UNIT_ASSERT_EQUAL(TStringType::Join(TStringType(Data.a()), static_cast<typename TStringType::char_type>('b'), TStringType(Data.cd()), TStringType(Data.e()), Data.fg(), TStringType(Data.h())), Data.abcdefgh());
    }

    void TestCopy() {
        TStringType s(Data.abcd());
        TStringType c = s.copy();

        UNIT_ASSERT_EQUAL(s, c);
        UNIT_ASSERT(s.end() != c.end());
    }

    void TestStrCpy() {
        {
            TStringType s(Data.abcd());
            char_type data[5];

            data[4] = 1;

            s.strcpy(data, 4);

            UNIT_ASSERT_EQUAL(data[0], *Data.a());
            UNIT_ASSERT_EQUAL(data[1], *Data.b());
            UNIT_ASSERT_EQUAL(data[2], *Data.c());
            UNIT_ASSERT_EQUAL(data[3], 0);
            UNIT_ASSERT_EQUAL(data[4], 1);
        }

        {
            TStringType s(Data.abcd());
            char_type data[5];

            s.strcpy(data, 5);

            UNIT_ASSERT_EQUAL(data[0], *Data.a());
            UNIT_ASSERT_EQUAL(data[1], *Data.b());
            UNIT_ASSERT_EQUAL(data[2], *Data.c());
            UNIT_ASSERT_EQUAL(data[3], *Data.d());
            UNIT_ASSERT_EQUAL(data[4], 0);
        }
    }

    void TestPrefixSuffix() {
        const TStringType emptyStr;
        UNIT_ASSERT_EQUAL(emptyStr.StartsWith('x'), false);
        UNIT_ASSERT_EQUAL(emptyStr.EndsWith('x'), false);
        UNIT_ASSERT_EQUAL(emptyStr.StartsWith(0), false);
        UNIT_ASSERT_EQUAL(emptyStr.EndsWith(0), false);
        UNIT_ASSERT_EQUAL(emptyStr.StartsWith(emptyStr), true);
        UNIT_ASSERT_EQUAL(emptyStr.EndsWith(emptyStr), true);

        const char_type chars[] = {'h', 'e', 'l', 'l', 'o', 0};
        const TStringType str(chars);
        UNIT_ASSERT_EQUAL(str.StartsWith('h'), true);
        UNIT_ASSERT_EQUAL(str.StartsWith('o'), false);
        UNIT_ASSERT_EQUAL(str.EndsWith('o'), true);
        UNIT_ASSERT_EQUAL(str.EndsWith('h'), false);
        UNIT_ASSERT_EQUAL(str.StartsWith(emptyStr), true);
        UNIT_ASSERT_EQUAL(str.EndsWith(emptyStr), true);
    }

#ifndef TSTRING_IS_STD_STRING
    void TestCharRef() {
        const char_type abc[] = {'a', 'b', 'c', 0};
        const char_type bbc[] = {'b', 'b', 'c', 0};
        const char_type cbc[] = {'c', 'b', 'c', 0};

        TStringType s0 = abc;
        TStringType s1 = s0;

        UNIT_ASSERT(!s0.IsDetached());
        UNIT_ASSERT(!s1.IsDetached());

        /* Read access shouldn't detach. */
        UNIT_ASSERT_VALUES_EQUAL(s0[0], (ui8)'a');
        UNIT_ASSERT(!s0.IsDetached());
        UNIT_ASSERT(!s1.IsDetached());

        /* Writing should detach. */
        s1[0] = (ui8)'b';
        TStringType s2 = s0;
        s0[0] = (ui8)'c';

        UNIT_ASSERT_VALUES_EQUAL(s0, cbc);
        UNIT_ASSERT_VALUES_EQUAL(s1, bbc);
        UNIT_ASSERT_VALUES_EQUAL(s2, abc);
        UNIT_ASSERT(s0.IsDetached());
        UNIT_ASSERT(s1.IsDetached());
        UNIT_ASSERT(s2.IsDetached());

        /* Accessing null terminator is OK. Note that writing into it is UB. */
        UNIT_ASSERT_VALUES_EQUAL(s0[3], (ui8)'\0');
        UNIT_ASSERT_VALUES_EQUAL(s1[3], (ui8)'\0');
        UNIT_ASSERT_VALUES_EQUAL(s2[3], (ui8)'\0');

        /* Assignment one char reference to another results in modification of underlying character */
        {
            const char_type dark_eyed[] = {'d', 'a', 'r', 'k', '-', 'e', 'y', 'e', 'd', 0};
            const char_type red_eared[] = {'r', 'e', 'd', '-', 'e', 'a', 'r', 'e', 'd', 0};
            TStringType s0 = dark_eyed;
            TStringType s1 = TStringType::Uninitialized(s0.size());
            for (size_t i = 0; i < s1.size(); ++i) {
                const size_t j = (3u * (i + 1u) ^ 1u) % s0.size();
                s1[i] = s0[j];
            }
            UNIT_ASSERT_VALUES_EQUAL(s1, red_eared);
        }
    }
#endif

    void TestBack() {
        const char_type chars[] = {'f', 'o', 'o', 0};

        TStringType str = chars;
        const TStringType constStr = str;

        UNIT_ASSERT_VALUES_EQUAL(constStr.back(), (ui8)'o');
        UNIT_ASSERT_VALUES_EQUAL(str.back(), (ui8)'o');

        str.back() = 'r';
        UNIT_ASSERT_VALUES_EQUAL(constStr.back(), (ui8)'o');
        UNIT_ASSERT_VALUES_EQUAL(str.back(), (ui8)'r');
    }

    void TestFront() {
        const char_type chars[] = {'f', 'o', 'o', 0};

        TStringType str = chars;
        const TStringType constStr = str;

        UNIT_ASSERT_VALUES_EQUAL(constStr.front(), (ui8)'f');
        UNIT_ASSERT_VALUES_EQUAL(str.front(), (ui8)'f');

        str.front() = 'r';
        UNIT_ASSERT_VALUES_EQUAL(constStr.front(), (ui8)'f');
        UNIT_ASSERT_VALUES_EQUAL(str.front(), (ui8)'r');
    }

    void TestIterators() {
        const char_type chars[] = {'f', 'o', 0};

        TStringType str = chars;
        const TStringType constStr = str;

        typename TStringType::const_iterator itBegin = str.begin();
        typename TStringType::const_iterator itEnd = str.end();
        typename TStringType::const_iterator citBegin = constStr.begin();
        typename TStringType::const_iterator citEnd = constStr.end();

        UNIT_ASSERT_VALUES_EQUAL(*itBegin, (ui8)'f');
        UNIT_ASSERT_VALUES_EQUAL(*citBegin, (ui8)'f');

        str.front() = 'r';
        UNIT_ASSERT_VALUES_EQUAL(*itBegin, (ui8)'r');
        UNIT_ASSERT_VALUES_EQUAL(*citBegin, (ui8)'f');

        UNIT_ASSERT_VALUES_EQUAL(2, itEnd - itBegin);
        UNIT_ASSERT_VALUES_EQUAL(2, citEnd - citBegin);

        UNIT_ASSERT_VALUES_EQUAL(*(++itBegin), (ui8)'o');
        UNIT_ASSERT_VALUES_EQUAL(*(++citBegin), (ui8)'o');

        UNIT_ASSERT_VALUES_EQUAL(*(--itBegin), (ui8)'r');
        UNIT_ASSERT_VALUES_EQUAL(*(--citBegin), (ui8)'f');

        UNIT_ASSERT_VALUES_EQUAL(*(itBegin++), (ui8)'r');
        UNIT_ASSERT_VALUES_EQUAL(*(citBegin++), (ui8)'f');
        UNIT_ASSERT_VALUES_EQUAL(*itBegin, (ui8)'o');
        UNIT_ASSERT_VALUES_EQUAL(*citBegin, (ui8)'o');

        UNIT_ASSERT_VALUES_EQUAL(*(itBegin--), (ui8)'o');
        UNIT_ASSERT_VALUES_EQUAL(*(citBegin--), (ui8)'o');
        UNIT_ASSERT_VALUES_EQUAL(*itBegin, (ui8)'r');
        UNIT_ASSERT_VALUES_EQUAL(*citBegin, (ui8)'f');
    }

    void TestReverseIterators() {
        const char_type chars[] = {'f', 'o', 0};

        TStringType str = chars;
        const TStringType constStr = str;

        typename TStringType::reverse_iterator ritBegin = str.rbegin();
        typename TStringType::reverse_iterator ritEnd = str.rend();
        typename TStringType::const_reverse_iterator critBegin = constStr.rbegin();
        typename TStringType::const_reverse_iterator critEnd = constStr.rend();

        UNIT_ASSERT_VALUES_EQUAL(*ritBegin, (ui8)'o');
        UNIT_ASSERT_VALUES_EQUAL(*critBegin, (ui8)'o');

        str.back() = (ui8)'r';
        UNIT_ASSERT_VALUES_EQUAL(*ritBegin, (ui8)'r');
        UNIT_ASSERT_VALUES_EQUAL(*critBegin, (ui8)'o');

        UNIT_ASSERT_VALUES_EQUAL(2, ritEnd - ritBegin);
        UNIT_ASSERT_VALUES_EQUAL(2, critEnd - critBegin);

        UNIT_ASSERT_VALUES_EQUAL(*(++ritBegin), (ui8)'f');
        UNIT_ASSERT_VALUES_EQUAL(*(++critBegin), (ui8)'f');

        UNIT_ASSERT_VALUES_EQUAL(*(--ritBegin), (ui8)'r');
        UNIT_ASSERT_VALUES_EQUAL(*(--critBegin), (ui8)'o');

        UNIT_ASSERT_VALUES_EQUAL(*(ritBegin++), (ui8)'r');
        UNIT_ASSERT_VALUES_EQUAL(*(critBegin++), (ui8)'o');
        UNIT_ASSERT_VALUES_EQUAL(*ritBegin, (ui8)'f');
        UNIT_ASSERT_VALUES_EQUAL(*critBegin, (ui8)'f');

        UNIT_ASSERT_VALUES_EQUAL(*(ritBegin--), (ui8)'f');
        UNIT_ASSERT_VALUES_EQUAL(*(critBegin--), (ui8)'f');
        UNIT_ASSERT_VALUES_EQUAL(*ritBegin, (ui8)'r');
        UNIT_ASSERT_VALUES_EQUAL(*critBegin, (ui8)'o');

        *ritBegin = (ui8)'e';
        UNIT_ASSERT_VALUES_EQUAL(*ritBegin, (ui8)'e');

        str = chars;
        auto it = std::find_if(
            str.rbegin(), str.rend(),
            [](char_type c) { return c == 'o'; });
        UNIT_ASSERT_EQUAL(it, str.rbegin());
    }
};
