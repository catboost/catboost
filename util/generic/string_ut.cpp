#include "string.h"
#include "deque.h"
#include "vector.h"
#include "yexception.h"
#include "strbuf.h"

#include <library/cpp/unittest/registar.h>

#include <util/string/subst.h>
#include <util/stream/output.h>
#include <util/charset/wide.h>
#include <util/str_stl.h>

#include <string>
#include <sstream>
#include <algorithm>
#include <stdexcept>

static_assert(sizeof(TString) == sizeof(const char*), "expect sizeof(TString) == sizeof(const char*)");

template <class TStringType, typename TTestData>
class TStringTestImpl {
protected:
    using char_type = typename TStringType::char_type;
    using traits_type = typename TStringType::traits_type;

    TTestData Data;

public:
    void TestMaxSize() {
        size_t l = TStringType::TDataTraits::MaxSize;
        UNIT_ASSERT(TStringType::TDataTraits::CalcAllocationSizeAndCapacity(l) >= TStringType::TDataTraits::MaxSize * sizeof(char_type));
        UNIT_ASSERT(l >= TStringType::TDataTraits::MaxSize);

        const size_t badMaxVal = TStringType::TDataTraits::MaxSize + 1;
        l = badMaxVal;
        UNIT_ASSERT(TStringType::TDataTraits::CalcAllocationSizeAndCapacity(l) < badMaxVal * sizeof(char_type));

        TStringType s;
        UNIT_CHECK_GENERATED_EXCEPTION(s.reserve(badMaxVal), std::length_error);
    }

    void TestConstructors() {
        TStringType s0(nullptr);
        UNIT_ASSERT(s0.size() == 0);
        UNIT_ASSERT_EQUAL(s0, TStringType());

        TStringType s;
        TStringType s1(*Data._0());
        TStringType s2(Data._0());
        UNIT_ASSERT(s1 == s2);

        TStringType s3 = TStringType::Uninitialized(10);
        UNIT_ASSERT(s3.size() == 10);

        TStringType s4(Data._0123456(), 1, 3);
        UNIT_ASSERT(s4 == Data._123());

        TStringType s5(5, *Data._0());
        UNIT_ASSERT(s5 == Data._00000());

        TStringType s6(Data._0123456());
        UNIT_ASSERT(s6 == Data._0123456());
        TStringType s7(s6);
        UNIT_ASSERT(s7 == s6);
        UNIT_ASSERT(s7.c_str() == s6.c_str());

        TStringType s8(s7, 1, 3);
        UNIT_ASSERT(s8 == Data._123());

        TStringType s9(*Data._1());
        UNIT_ASSERT(s9 == Data._1());

        TStringType s10(Reserve(100));
        UNIT_ASSERT(s10.empty());
        UNIT_ASSERT(s10.capacity() > 100);
    }

    void TestReplace() {
        TStringType s(Data._0123456());
        UNIT_ASSERT(s.copy() == Data._0123456());

        // append family
        s.append(Data.x());
        UNIT_ASSERT(s == Data._0123456x());

        s.append(Data.xyz(), 1, 1);
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

    void TestRefCount() {
        // access protected member
        class TestStroka: public TStringType {
        public:
            TestStroka(const char_type* s)
                : TStringType(s)
            {
            }
            intptr_t RefCount() const {
                return TStringType::GetData()->Refs;
            }
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

    //  Find family

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

        UNIT_ASSERT(s > TBasicStringBuf<char_type>(Data.abc0123456xyz()));
        UNIT_ASSERT(TBasicStringBuf<char_type>(Data.abc0123456xyz()) < s);
        UNIT_ASSERT(s == TBasicStringBuf<char_type>(Data.abcd()));
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
        UNIT_ASSERT(s.length() == traits_type::GetLength(s.data()));

        // is_null()
        TStringType s1(Data.Empty());
        UNIT_ASSERT(s1.is_null() == true);
        UNIT_ASSERT(s1.is_null() == s1.empty());
        UNIT_ASSERT(s1.is_null() == !s1);

        TStringType s2(s);
        UNIT_ASSERT(s2 == s);

        // reverse()
        s2.reverse();
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

        sS.hash(); /*size_t hash_val = sS.hash();

        try {
            //UNIT_ASSERT(hash_val == Data.HashOf_0123456());
        } catch (...) {
            Cerr << hash_val << Endl;
            throw;
        }*/

        s2.assign(Data._0123456(), 2, 2);
        UNIT_ASSERT(s2 == Data._23());

        //s2.reserve();

        TStringType s5(Data.abcde());
        s5 = nullptr;
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
    }
};

class TStringTestZero: public TTestBase {
    UNIT_TEST_SUITE(TStringTestZero);
    UNIT_TEST(TestZero);
    UNIT_TEST_SUITE_END();

public:
    void TestZero() {
        const char data[] = "abc\0def\0";
        TString s(data, sizeof(data));
        UNIT_ASSERT(s.size() == sizeof(data));
        UNIT_ASSERT(s.StartsWith(s));
        UNIT_ASSERT(s.EndsWith(s));
        UNIT_ASSERT(s.Contains('\0'));

        const char raw_def[] = "def";
        const char raw_zero[] = "\0";
        TString def(raw_def, sizeof(raw_def) - 1);
        TString zero(raw_zero, sizeof(raw_zero) - 1);
        UNIT_ASSERT_EQUAL(4, s.find(raw_def));
        UNIT_ASSERT_EQUAL(4, s.find(def));
        UNIT_ASSERT_EQUAL(4, s.find_first_of(raw_def));
        UNIT_ASSERT_EQUAL(3, s.find_first_of(zero));
        UNIT_ASSERT_EQUAL(7, s.find_first_not_of(def, 4));

        const char nonSubstring[] = "def\0ghi";
        UNIT_ASSERT_EQUAL(TString::npos, s.find(TString(nonSubstring, sizeof(nonSubstring))));

        TString copy = s;
        copy.replace(copy.size() - 1, 1, "z");
        UNIT_ASSERT(s != copy);
        copy.replace(copy.size() - 1, 1, "\0", 0, 1);
        UNIT_ASSERT(s == copy);

        TString prefix(data, 5);
        UNIT_ASSERT(s.StartsWith(prefix));
        UNIT_ASSERT(s != prefix);
        UNIT_ASSERT(s > prefix);
        UNIT_ASSERT(s > s.data());
        UNIT_ASSERT(s == TString(s.data(), s.size()));
        UNIT_ASSERT(data < s);

        s.remove(5);
        UNIT_ASSERT(s == prefix);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TStringTestZero);

template <typename TStringType, typename TTestData>
class TStringStdTestImpl {
    using char_type = typename TStringType::char_type;
    using traits_type = typename TStringType::traits_type;

    TTestData Data;

protected:
    void constructor() {
        // @todo use UNIT_TEST_EXCEPTION
        try {
            TStringType s((size_t)-1, *Data.a());
            UNIT_ASSERT(false);
        } catch (const std::length_error&) {
            UNIT_ASSERT(true);
        } catch (...) {
            //Expected exception is length_error:
            UNIT_ASSERT(false);
        }
    }

    void reserve() {
        TStringType s;
        // @todo use UNIT_TEST_EXCEPTION
        try {
            s.reserve(s.max_size() + 1);
            UNIT_ASSERT(false);
        } catch (const std::length_error&) {
            UNIT_ASSERT(true);
        } catch (...) {
            //Expected exception is length_error:
            UNIT_ASSERT(false);
        }

        // Non-shared behaviour - never shrink

        s.reserve(256);
        const auto* data = s.data();

        UNIT_ASSERT(s.capacity() >= 256);

        s.reserve(128);

        UNIT_ASSERT(s.capacity() >= 256 && s.data() == data);

        s.resize(64, 'x');
        s.reserve(10);

        UNIT_ASSERT(s.capacity() >= 256 && s.data() == data);

        // Shared behaviour - always reallocate, just as much as requisted

        TStringType holder = s;

        UNIT_ASSERT(s.capacity() >= 256);

        s.reserve(128);

        UNIT_ASSERT(s.capacity() >= 128 && s.capacity() < 256 && s.data() != data);
        UNIT_ASSERT(s.IsDetached());

        s.resize(64, 'x');
        data = s.data();
        holder = s;

        s.reserve(10);

        UNIT_ASSERT(s.capacity() >= 64 && s.capacity() < 128 && s.data() != data);
        UNIT_ASSERT(s.IsDetached());
    }

    void short_string() {
        TStringType const ref_short_str1(Data.str1()), ref_short_str2(Data.str2());
        TStringType short_str1(ref_short_str1), short_str2(ref_short_str2);
        TStringType const ref_long_str1(Data.str__________________________________________________1());
        TStringType const ref_long_str2(Data.str__________________________________________________2());
        TStringType long_str1(ref_long_str1), long_str2(ref_long_str2);

        UNIT_ASSERT(short_str1 == ref_short_str1);
        UNIT_ASSERT(long_str1 == ref_long_str1);

        {
            TStringType str1(short_str1);
            str1 = long_str1;
            UNIT_ASSERT(str1 == ref_long_str1);
        }

        {
            TStringType str1(long_str1);
            str1 = short_str1;
            UNIT_ASSERT(str1 == ref_short_str1);
        }

        {
            short_str1.swap(short_str2);
            UNIT_ASSERT((short_str1 == ref_short_str2) && (short_str2 == ref_short_str1));
            short_str1.swap(short_str2);
        }

        {
            long_str1.swap(long_str2);
            UNIT_ASSERT((long_str1 == ref_long_str2) && (long_str2 == ref_long_str1));
            long_str1.swap(long_str2);
        }

        {
            short_str1.swap(long_str1);
            UNIT_ASSERT((short_str1 == ref_long_str1) && (long_str1 == ref_short_str1));
            short_str1.swap(long_str1);
        }

        {
            long_str1.swap(short_str1);
            UNIT_ASSERT((short_str1 == ref_long_str1) && (long_str1 == ref_short_str1));
            long_str1.swap(short_str1);
        }

        {
            //This is to test move constructor
            TVector<TStringType> str_vect;

            str_vect.push_back(short_str1);
            str_vect.push_back(long_str1);
            str_vect.push_back(short_str2);
            str_vect.push_back(long_str2);

            UNIT_ASSERT(
                (str_vect[0] == ref_short_str1) &&
                (str_vect[1] == ref_long_str1) &&
                (str_vect[2] == ref_short_str2) &&
                (str_vect[3] == ref_long_str2));
        }
    }

    void erase() {
        char_type const* c_str = Data.Hello_World();
        TStringType str(c_str);
        UNIT_ASSERT(str == c_str);

        str.erase(str.begin() + 1, str.end() - 1); // Erase all but first and last.

        size_t i;
        for (i = 0; i < str.size(); ++i) {
            switch (i) {
                case 0:
                    UNIT_ASSERT(str[i] == *Data.H());
                    break;

                case 1:
                    UNIT_ASSERT(str[i] == *Data.d());
                    break;

                default:
                    UNIT_ASSERT(false);
            }
        }

        str.insert(1, c_str);
        str.erase(str.begin());   // Erase first element.
        str.erase(str.end() - 1); // Erase last element.
        UNIT_ASSERT(str == c_str);
        str.clear(); // Erase all.
        UNIT_ASSERT(str.empty());

        str = c_str;
        UNIT_ASSERT(str == c_str);

        str.erase(1, str.size() - 1); // Erase all but first and last.
        for (i = 0; i < str.size(); i++) {
            switch (i) {
                case 0:
                    UNIT_ASSERT(str[i] == *Data.H());
                    break;

                case 1:
                    UNIT_ASSERT(str[i] == *Data.d());
                    break;

                default:
                    UNIT_ASSERT(false);
            }
        }

        str.erase(1);
        UNIT_ASSERT(str == Data.H());
    }

    void data() {
        TStringType xx;

        // ISO-IEC-14882:1998(E), 21.3.6, paragraph 3
        UNIT_ASSERT(xx.data() != nullptr);
    }

    void c_str() {
        TStringType low(Data._2004_01_01());
        TStringType xx;
        TStringType yy;

        // ISO-IEC-14882:1998(E), 21.3.6, paragraph 1
        UNIT_ASSERT(*(yy.c_str()) == 0);

        // Blocks A and B should follow each other.
        // Block A:
        xx = Data._123456();
        xx += low;
        UNIT_ASSERT(traits_type::Compare(xx.c_str(), Data._1234562004_01_01()) == 0);
        // End of block A

        // Block B:
        xx = Data._1234();
        xx += Data._5();
        UNIT_ASSERT(traits_type::Compare(xx.c_str(), Data._12345()) == 0);
        // End of block B
    }

    void null_char() {
        // ISO/IEC 14882:1998(E), ISO/IEC 14882:2003(E), 21.3.4 ('... the const version')
        const TStringType s(Data._123456());

        UNIT_ASSERT(s[s.size()] == 0);

#if 0
        try {
            //Check is only here to avoid warning about value of expression not used
            UNIT_ASSERT(s.at(s.size()) == 0);
            UNIT_ASSERT(false);
        } catch (const std::out_of_range&) {
            UNIT_ASSERT(true);
        } catch (...) {
            UNIT_ASSERT(false);
        }
#endif
    }

    void insert() {
        TStringType strorg = Data.This_is_test_string_for_string_calls();
        TStringType str;

        // In case of reallocation there is no auto reference problem
        // so we reserve a big enough TStringType to be sure to test this
        // particular point.

        str.reserve(100);
        str = strorg;

        //test self insertion:
        str.insert(10, str.c_str() + 5, 15);
        UNIT_ASSERT(str == Data.This_is_teis_test_string_st_string_for_string_calls());

        str = strorg;
        str.insert(15, str.c_str() + 5, 25);
        UNIT_ASSERT(str == Data.This_is_test_stis_test_string_for_stringring_for_string_calls());

        str = strorg;
        str.insert(0, str.c_str() + str.size() - 4, 4);
        UNIT_ASSERT(str == Data.allsThis_is_test_string_for_string_calls());

        str = strorg;
        str.insert(0, str.c_str() + str.size() / 2 - 1, str.size() / 2 + 1);
        UNIT_ASSERT(str == Data.ng_for_string_callsThis_is_test_string_for_string_calls());

        str = strorg;
        typename TStringType::iterator b = str.begin();
        typename TStringType::const_iterator s = str.begin() + str.size() / 2 - 1;
        typename TStringType::const_iterator e = str.end();
        str.insert(b, s, e);
        UNIT_ASSERT(str == Data.ng_for_string_callsThis_is_test_string_for_string_calls());

#if 0
        // AV
        str = strorg;
        str.insert(str.begin(), str.begin() + str.size() / 2 - 1, str.end());
        UNIT_ASSERT(str == Data.ng_for_string_callsThis_is_test_string_for_string_calls());
#endif

        TStringType str0;
        str0.insert(str0.begin(), 5, *Data._0());
        UNIT_ASSERT(str0 == Data._00000());

        TStringType str1;
        {
            typename TStringType::size_type pos = 0, nb = 2;
            str1.insert(pos, nb, *Data._1());
        }
        UNIT_ASSERT(str1 == Data._11());

        str0.insert(0, str1);
        UNIT_ASSERT(str0 == Data._1100000());

        TStringType str2(Data._2345());
        str0.insert(str0.size(), str2, 1, 2);
        UNIT_ASSERT(str0 == Data._110000034());

        str1.insert(str1.begin() + 1, 2, *Data._2());
        UNIT_ASSERT(str1 == Data._1221());

        str1.insert(2, Data._333333(), 3);
        UNIT_ASSERT(str1 == Data._1233321());

        str1.insert(4, Data._4444());
        UNIT_ASSERT(str1 == Data._12334444321());

        str1.insert(str1.begin() + 6, *Data._5());
        UNIT_ASSERT(str1 == Data._123344544321());
    }

    void resize() {
        TStringType s;

        s.resize(0);

        UNIT_ASSERT(*s.c_str() == 0);

        s = Data._1234567();

        s.resize(0);
        UNIT_ASSERT(*s.c_str() == 0);

        s = Data._1234567();
        s.resize(1);
        UNIT_ASSERT(s.size() == 1);
        UNIT_ASSERT(*s.c_str() == *Data._1());
        UNIT_ASSERT(*(s.c_str() + 1) == 0);

        s = Data._1234567();
#if 0
        s.resize(10);
#else
        s.resize(10, 0);
#endif
        UNIT_ASSERT(s.size() == 10);
        UNIT_ASSERT(s[6] == *Data._7());
        UNIT_ASSERT(s[7] == 0);
        UNIT_ASSERT(s[8] == 0);
        UNIT_ASSERT(s[9] == 0);
    }

    void find() {
        TStringType s(Data.one_two_three_one_two_three());

        UNIT_ASSERT(s.find(Data.one()) == 0);
        UNIT_ASSERT(s.find(*Data.t()) == 4);
        UNIT_ASSERT(s.find(*Data.t(), 5) == 8);

        UNIT_ASSERT(s.find(Data.four()) == TStringType::npos);
        UNIT_ASSERT(s.find(Data.one(), TStringType::npos) == TStringType::npos);
        UNIT_ASSERT(s.find_first_of(Data.abcde()) == 2);
        UNIT_ASSERT(s.find_first_not_of(Data.enotw_()) == 9);
    }

    void capacity() {
        TStringType s;

        UNIT_ASSERT(s.capacity() < s.max_size());
        UNIT_ASSERT(s.capacity() >= s.size());

        for (int i = 0; i < 18; ++i) {
            s += ' ';

            UNIT_ASSERT(s.capacity() > 0);
            UNIT_ASSERT(s.capacity() < s.max_size());
            UNIT_ASSERT(s.capacity() >= s.size());
        }
    }

    void assign() {
        TStringType s;
        char_type const* cstr = Data.test_string_for_assign();

        s.assign(cstr, cstr + 22);
        UNIT_ASSERT(s == Data.test_string_for_assign());

        TStringType s2(Data.other_test_string());
        s.assign(s2);
        UNIT_ASSERT(s == s2);

        static TStringType str1;
        static TStringType str2;

        // short TStringType optim:
        str1 = Data._123456();
        // longer than short TStringType:
        str2 = Data._1234567890123456789012345678901234567890();

        UNIT_ASSERT(str1[5] == *Data._6());
        UNIT_ASSERT(str2[29] == *Data._0());
    }

    void copy() {
        TStringType s(Data.foo());
        char_type dest[4];
        dest[0] = dest[1] = dest[2] = dest[3] = 1;
        s.copy(dest, 4);
        int pos = 0;
        UNIT_ASSERT(dest[pos++] == *Data.f());
        UNIT_ASSERT(dest[pos++] == *Data.o());
        UNIT_ASSERT(dest[pos++] == *Data.o());
        UNIT_ASSERT(dest[pos++] == 1);

        dest[0] = dest[1] = dest[2] = dest[3] = 1;
        s.copy(dest, 4, 2);
        pos = 0;
        UNIT_ASSERT(dest[pos++] == *Data.o());
        UNIT_ASSERT(dest[pos++] == 1);

        // @todo use UNIT_TEST_EXCEPTION
        try {
            s.copy(dest, 4, 5);
            UNIT_ASSERT(!"expected std::out_of_range");
        } catch (const std::out_of_range&) {
            UNIT_ASSERT(true);
        } catch (...) {
            UNIT_ASSERT(!"expected std::out_of_range");
        }
    }

    void cbegin_cend() {
        const char helloThere[] = "Hello there";
        TString s = helloThere;
        size_t index = 0;
        for (auto it = s.cbegin(); s.cend() != it; ++it, ++index) {
            UNIT_ASSERT_VALUES_EQUAL(helloThere[index], *it);
        }
    }

    void compare() {
        TStringType str1(Data.abcdef());
        TStringType str2;

        str2 = Data.abcdef();
        UNIT_ASSERT(str1.compare(str2) == 0);
        UNIT_ASSERT(str1.compare(str2.data(), str2.size()) == 0);
        str2 = Data.abcde();
        UNIT_ASSERT(str1.compare(str2) > 0);
        UNIT_ASSERT(str1.compare(str2.data(), str2.size()) > 0);
        str2 = Data.abcdefg();
        UNIT_ASSERT(str1.compare(str2) < 0);
        UNIT_ASSERT(str1.compare(str2.data(), str2.size()) < 0);

        UNIT_ASSERT(str1.compare(Data.abcdef()) == 0);
        UNIT_ASSERT(str1.compare(Data.abcde()) > 0);
        UNIT_ASSERT(str1.compare(Data.abcdefg()) < 0);

        str2 = Data.cde();
        UNIT_ASSERT(str1.compare(2, 3, str2) == 0);
        str2 = Data.cd();
        UNIT_ASSERT(str1.compare(2, 3, str2) > 0);
        str2 = Data.cdef();
        UNIT_ASSERT(str1.compare(2, 3, str2) < 0);

        str2 = Data.abcdef();
        UNIT_ASSERT(str1.compare(2, 3, str2, 2, 3) == 0);
        UNIT_ASSERT(str1.compare(2, 3, str2, 2, 2) > 0);
        UNIT_ASSERT(str1.compare(2, 3, str2, 2, 4) < 0);

        UNIT_ASSERT(str1.compare(2, 3, Data.cdefgh(), 3) == 0);
        UNIT_ASSERT(str1.compare(2, 3, Data.cdefgh(), 2) > 0);
        UNIT_ASSERT(str1.compare(2, 3, Data.cdefgh(), 4) < 0);
    }

    void find_last_of() {
        // 21.3.6.4
        TStringType s(Data.one_two_three_one_two_three());

        UNIT_ASSERT(s.find_last_of(Data.abcde()) == 26);
        UNIT_ASSERT(s.find_last_of(TStringType(Data.abcde())) == 26);

        TStringType test(Data.aba());

        UNIT_ASSERT(test.find_last_of(Data.a(), 2, 1) == 2);
        UNIT_ASSERT(test.find_last_of(Data.a(), 1, 1) == 0);
        UNIT_ASSERT(test.find_last_of(Data.a(), 0, 1) == 0);

        UNIT_ASSERT(test.find_last_of(*Data.a(), 2) == 2);
        UNIT_ASSERT(test.find_last_of(*Data.a(), 1) == 0);
        UNIT_ASSERT(test.find_last_of(*Data.a(), 0) == 0);
    }
#if 0
    void rfind() {
        // 21.3.6.2
        TStringType s(Data.one_two_three_one_two_three());

        UNIT_ASSERT(s.rfind(Data.two()) == 18);
        UNIT_ASSERT(s.rfind(Data.two(), 0) == TStringType::npos);
        UNIT_ASSERT(s.rfind(Data.two(), 11) == 4);
        UNIT_ASSERT(s.rfind(*Data.w()) == 19);

        TStringType test(Data.aba());

        UNIT_ASSERT(test.rfind(Data.a(), 2, 1) == 2);
        UNIT_ASSERT(test.rfind(Data.a(), 1, 1) == 0);
        UNIT_ASSERT(test.rfind(Data.a(), 0, 1) == 0);

        UNIT_ASSERT(test.rfind(*Data.a(), 2) == 2);
        UNIT_ASSERT(test.rfind(*Data.a(), 1) == 0);
        UNIT_ASSERT(test.rfind(*Data.a(), 0) == 0);
    }
#endif
    void find_last_not_of() {
        // 21.3.6.6
        TStringType s(Data.one_two_three_one_two_three());

        UNIT_ASSERT(s.find_last_not_of(Data.ehortw_()) == 15);

        TStringType test(Data.aba());

        UNIT_ASSERT(test.find_last_not_of(Data.a(), 2, 1) == 1);
        UNIT_ASSERT(test.find_last_not_of(Data.b(), 2, 1) == 2);
        UNIT_ASSERT(test.find_last_not_of(Data.a(), 1, 1) == 1);
        UNIT_ASSERT(test.find_last_not_of(Data.b(), 1, 1) == 0);
        UNIT_ASSERT(test.find_last_not_of(Data.a(), 0, 1) == TStringType::npos);
        UNIT_ASSERT(test.find_last_not_of(Data.b(), 0, 1) == 0);

        UNIT_ASSERT(test.find_last_not_of(*Data.a(), 2) == 1);
        UNIT_ASSERT(test.find_last_not_of(*Data.b(), 2) == 2);
        UNIT_ASSERT(test.find_last_not_of(*Data.a(), 1) == 1);
        UNIT_ASSERT(test.find_last_not_of(*Data.b(), 1) == 0);
        UNIT_ASSERT(test.find_last_not_of(*Data.a(), 0) == TStringType::npos);
        UNIT_ASSERT(test.find_last_not_of(*Data.b(), 0) == 0);
    }
#if 0
    void replace() {
        // This test case is for the non template basic_TString::replace method,
        // this is why we play with the const iterators and reference to guaranty
        // that the right method is called.

        const TStringType v(Data._78());
        TStringType s(Data._123456());
        TStringType const& cs = s;

        typename TStringType::iterator i = s.begin() + 1;
        s.replace(i, i + 3, v.begin(), v.end());
        UNIT_ASSERT(s == Data._17856());

        s = Data._123456();
        i = s.begin() + 1;
        s.replace(i, i + 1, v.begin(), v.end());
        UNIT_ASSERT(s == Data._1783456());

        s = Data._123456();
        i = s.begin() + 1;
        typename TStringType::const_iterator ci = s.begin() + 1;
        s.replace(i, i + 3, ci + 3, cs.end());
        UNIT_ASSERT(s == Data._15656());

        s = Data._123456();
        i = s.begin() + 1;
        ci = s.begin() + 1;
        s.replace(i, i + 3, ci, ci + 2);
        UNIT_ASSERT(s == Data._12356());

        s = Data._123456();
        i = s.begin() + 1;
        ci = s.begin() + 1;
        s.replace(i, i + 3, ci + 1, cs.end());
        UNIT_ASSERT(s == Data._1345656());

        s = Data._123456();
        i = s.begin();
        ci = s.begin() + 1;
        s.replace(i, i, ci, ci + 1);
        UNIT_ASSERT(s == Data._2123456());

        s = Data._123456();
        s.replace(s.begin() + 4, s.end(), cs.begin(), cs.end());
        UNIT_ASSERT(s == Data._1234123456());

        // This is the test for the template replace method.

        s = Data._123456();
        typename TStringType::iterator b = s.begin() + 4;
        typename TStringType::iterator e = s.end();
        typename TStringType::const_iterator rb = s.begin();
        typename TStringType::const_iterator re = s.end();
        s.replace(b, e, rb, re);
        UNIT_ASSERT(s == Data._1234123456());

        s = Data._123456();
        s.replace(s.begin() + 4, s.end(), s.begin(), s.end());
        UNIT_ASSERT(s == Data._1234123456());

        TStringType strorg(Data.This_is_test_StringT_for_StringT_calls());
        TStringType str = strorg;
        str.replace(5, 15, str.c_str(), 10);
        UNIT_ASSERT(str == Data.This_This_is_tefor_StringT_calls());

        str = strorg;
        str.replace(5, 5, str.c_str(), 10);
        UNIT_ASSERT(str == Data.This_This_is_test_StringT_for_StringT_calls());

#if !defined(STLPORT) || defined(_STLP_MEMBER_TEMPLATES)
        deque<char_type> cdeque;
        cdeque.push_back(*Data.I());
        str.replace(str.begin(), str.begin() + 11, cdeque.begin(), cdeque.end());
        UNIT_ASSERT(str == Data.Is_test_StringT_for_StringT_calls());
#endif
    }
#endif
}; // TStringStdTestImpl

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

class TStringTest: public TTestBase, private TStringTestImpl<TString, TTestData<char>> {
public:
    UNIT_TEST_SUITE(TStringTest);
    UNIT_TEST(TestMaxSize);
    UNIT_TEST(TestConstructors);
    UNIT_TEST(TestReplace);
    UNIT_TEST(TestRefCount);
    UNIT_TEST(TestFind);
    UNIT_TEST(TestContains);
    UNIT_TEST(TestOperators);
    UNIT_TEST(TestMulOperators);
    UNIT_TEST(TestFuncs);
    UNIT_TEST(TestUtils);
    UNIT_TEST(TestEmpty);
    UNIT_TEST(TestJoin);
    UNIT_TEST(TestCopy);
    UNIT_TEST(TestStrCpy);
    UNIT_TEST(TestPrefixSuffix);
    UNIT_TEST(TestCharRef);
    UNIT_TEST(TestBack)
    UNIT_TEST(TestFront)
    UNIT_TEST(TestIterators);
    UNIT_TEST(TestReverseIterators);
    //UNIT_TEST(TestOperatorsCI); must fail
    UNIT_TEST_SUITE_END();
};

UNIT_TEST_SUITE_REGISTRATION(TStringTest);

class TWideStringTest: public TTestBase, private TStringTestImpl<TUtf16String, TTestData<wchar16>> {
public:
    UNIT_TEST_SUITE(TWideStringTest);
    UNIT_TEST(TestConstructors);
    UNIT_TEST(TestReplace);
    UNIT_TEST(TestRefCount);
    UNIT_TEST(TestFind);
    UNIT_TEST(TestContains);
    UNIT_TEST(TestOperators);
    UNIT_TEST(TestLetOperator)
    UNIT_TEST(TestMulOperators);
    UNIT_TEST(TestFuncs);
    UNIT_TEST(TestUtils);
    UNIT_TEST(TestEmpty);
    UNIT_TEST(TestJoin);
    UNIT_TEST(TestCopy);
    UNIT_TEST(TestStrCpy);
    UNIT_TEST(TestPrefixSuffix);
    UNIT_TEST(TestCharRef);
    UNIT_TEST(TestBack);
    UNIT_TEST(TestFront)
    UNIT_TEST(TestDecodingMethods);
    UNIT_TEST(TestIterators);
    UNIT_TEST(TestReverseIterators);
    UNIT_TEST(TestStringLiterals);
    UNIT_TEST_SUITE_END();

private:
    void TestDecodingMethods() {
        UNIT_ASSERT(TUtf16String::FromAscii("").empty());
        UNIT_ASSERT(TUtf16String::FromAscii("abc") == ASCIIToWide("abc"));

        const char* text = "123kx83abcd ej)#$%ddja&%J&";
        TUtf16String wtext = ASCIIToWide(text);

        UNIT_ASSERT(wtext == TUtf16String::FromAscii(text));

        TString strtext(text);
        UNIT_ASSERT(wtext == TUtf16String::FromAscii(strtext));

        TStringBuf strbuftext(text);
        UNIT_ASSERT(wtext == TUtf16String::FromAscii(strbuftext));

        UNIT_ASSERT(wtext.substr(5) == TUtf16String::FromAscii(text + 5));

        const wchar16 wideCyrillicAlphabet[] = {
            0x0410, 0x0411, 0x0412, 0x0413, 0x0414, 0x0415, 0x0416, 0x0417, 0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E, 0x041F,
            0x0420, 0x0421, 0x0422, 0x0423, 0x0424, 0x0425, 0x0426, 0x0427, 0x0428, 0x0429, 0x042A, 0x042B, 0x042C, 0x042D, 0x042E, 0x042F,
            0x0430, 0x0431, 0x0432, 0x0433, 0x0434, 0x0435, 0x0436, 0x0437, 0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E, 0x043F,
            0x0440, 0x0441, 0x0442, 0x0443, 0x0444, 0x0445, 0x0446, 0x0447, 0x0448, 0x0449, 0x044A, 0x044B, 0x044C, 0x044D, 0x044E, 0x044F,
            0x00};

        TUtf16String strWide(wideCyrillicAlphabet);
        TString strUtf8 = WideToUTF8(strWide);

        UNIT_ASSERT(strWide == TUtf16String::FromUtf8(strUtf8.c_str()));
        UNIT_ASSERT(strWide == TUtf16String::FromUtf8(strUtf8));
        UNIT_ASSERT(strWide == TUtf16String::FromUtf8(TStringBuf(strUtf8)));

        // assign

        TUtf16String s1;
        s1.AssignAscii("1234");
        UNIT_ASSERT(s1 == ASCIIToWide("1234"));

        s1.AssignUtf8(strUtf8);
        UNIT_ASSERT(s1 == strWide);

        s1.AssignAscii(text);
        UNIT_ASSERT(s1 == wtext);

        // append

        TUtf16String s2;
        TUtf16String testAppend = strWide;
        s2.AppendUtf8(strUtf8);
        UNIT_ASSERT(testAppend == s2);

        testAppend += ' ';
        s2.AppendAscii(" ");
        UNIT_ASSERT(testAppend == s2);

        testAppend += '_';
        s2.AppendUtf8("_");
        UNIT_ASSERT(testAppend == s2);

        testAppend += wtext;
        s2.AppendAscii(text);
        UNIT_ASSERT(testAppend == s2);

        testAppend += wtext;
        s2.AppendUtf8(text);
        UNIT_ASSERT(testAppend == s2);
    }

    void TestLetOperator() {
        TUtf16String str;

        str = wchar16('X');
        UNIT_ASSERT(str == TUtf16String::FromAscii("X"));

        const TUtf16String hello = TUtf16String::FromAscii("hello");
        str = hello.data();
        UNIT_ASSERT(str == hello);

        str = hello;
        UNIT_ASSERT(str == hello);
    }

    void TestStringLiterals() {
        TUtf16String s1 = u"hello";
        UNIT_ASSERT_VALUES_EQUAL(s1, TUtf16String::FromAscii("hello"));

        TUtf16String s2 = u"привет";
        UNIT_ASSERT_VALUES_EQUAL(s2, TUtf16String::FromUtf8("привет"));
    }
};

UNIT_TEST_SUITE_REGISTRATION(TWideStringTest);

class TUtf32StringTest: public TTestBase, private TStringTestImpl<TUtf32String, TTestData<wchar32>> {
public:
    UNIT_TEST_SUITE(TUtf32StringTest);
    UNIT_TEST(TestConstructors);
    UNIT_TEST(TestReplace);
    UNIT_TEST(TestRefCount);
    UNIT_TEST(TestFind);
    UNIT_TEST(TestContains);
    UNIT_TEST(TestOperators);
    UNIT_TEST(TestLetOperator)
    UNIT_TEST(TestMulOperators);
    UNIT_TEST(TestFuncs);
    UNIT_TEST(TestUtils);
    UNIT_TEST(TestEmpty);
    UNIT_TEST(TestJoin);
    UNIT_TEST(TestCopy);
    UNIT_TEST(TestStrCpy);
    UNIT_TEST(TestPrefixSuffix);
    UNIT_TEST(TestCharRef);
    UNIT_TEST(TestBack);
    UNIT_TEST(TestFront)
    UNIT_TEST(TestDecodingMethods);
    UNIT_TEST(TestDecodingMethodsMixedStr);
    UNIT_TEST(TestIterators);
    UNIT_TEST(TestReverseIterators);
    UNIT_TEST(TestStringLiterals);
    UNIT_TEST_SUITE_END();

private:
    void TestDecodingMethods() {
        UNIT_ASSERT(TUtf32String::FromAscii("").empty());
        UNIT_ASSERT(TUtf32String::FromAscii("abc") == ASCIIToUTF32("abc"));

        const char* text = "123kx83abcd ej)#$%ddja&%J&";
        TUtf32String wtext = ASCIIToUTF32(text);

        UNIT_ASSERT(wtext == TUtf32String::FromAscii(text));

        TString strtext(text);
        UNIT_ASSERT(wtext == TUtf32String::FromAscii(strtext));

        TStringBuf strbuftext(text);
        UNIT_ASSERT(wtext == TUtf32String::FromAscii(strbuftext));

        UNIT_ASSERT(wtext.substr(5) == TUtf32String::FromAscii(text + 5));

        const wchar32 wideCyrillicAlphabet[] = {
            0x0410, 0x0411, 0x0412, 0x0413, 0x0414, 0x0415, 0x0416, 0x0417, 0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E, 0x041F,
            0x0420, 0x0421, 0x0422, 0x0423, 0x0424, 0x0425, 0x0426, 0x0427, 0x0428, 0x0429, 0x042A, 0x042B, 0x042C, 0x042D, 0x042E, 0x042F,
            0x0430, 0x0431, 0x0432, 0x0433, 0x0434, 0x0435, 0x0436, 0x0437, 0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E, 0x043F,
            0x0440, 0x0441, 0x0442, 0x0443, 0x0444, 0x0445, 0x0446, 0x0447, 0x0448, 0x0449, 0x044A, 0x044B, 0x044C, 0x044D, 0x044E, 0x044F,
            0x00};

        TUtf32String strWide(wideCyrillicAlphabet);
        TString strUtf8 = WideToUTF8(strWide);

        UNIT_ASSERT(strWide == TUtf32String::FromUtf8(strUtf8.c_str()));
        UNIT_ASSERT(strWide == TUtf32String::FromUtf8(strUtf8));
        UNIT_ASSERT(strWide == TUtf32String::FromUtf8(TStringBuf(strUtf8)));

        // assign

        TUtf32String s1;
        s1.AssignAscii("1234");
        UNIT_ASSERT(s1 == ASCIIToUTF32("1234"));

        s1.AssignUtf8(strUtf8);
        UNIT_ASSERT(s1 == strWide);

        s1.AssignAscii(text);
        UNIT_ASSERT(s1 == wtext);

        // append

        TUtf32String s2;
        TUtf32String testAppend = strWide;
        s2.AppendUtf8(strUtf8);
        UNIT_ASSERT(testAppend == s2);

        testAppend += ' ';
        s2.AppendAscii(" ");
        UNIT_ASSERT(testAppend == s2);

        testAppend += '_';
        s2.AppendUtf8("_");
        UNIT_ASSERT(testAppend == s2);

        testAppend += wtext;
        s2.AppendAscii(text);
        UNIT_ASSERT(testAppend == s2);

        testAppend += wtext;
        s2.AppendUtf8(text);

        UNIT_ASSERT(testAppend == s2);
    }

    void TestDecodingMethodsMixedStr() {
        UNIT_ASSERT(TUtf32String::FromAscii("").empty());
        UNIT_ASSERT(TUtf32String::FromAscii("abc") == ASCIIToUTF32("abc"));

        const char* text = "123kx83abcd ej)#$%ddja&%J&";
        TUtf32String wtext = ASCIIToUTF32(text);

        UNIT_ASSERT(wtext == TUtf32String::FromAscii(text));

        TString strtext(text);
        UNIT_ASSERT(wtext == TUtf32String::FromAscii(strtext));

        TStringBuf strbuftext(text);
        UNIT_ASSERT(wtext == TUtf32String::FromAscii(strbuftext));

        UNIT_ASSERT(wtext.substr(5) == TUtf32String::FromAscii(text + 5));

        const wchar32 cyrilicAndLatinWide[] = {
            0x0410, 0x0411, 0x0412, 0x0413, 0x0414, 0x0415, 0x0416, 0x0417, 0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E, 0x041F,
            0x0420, 0x0421, 0x0422, 0x0423, 0x0424, 0x0425, 0x0426, 0x0427, 0x0428, 0x0429, 0x042A, 0x042B, 0x042C, 0x042D, 0x042E, 0x042F,
            0x0430, 0x0431, 0x0432, 0x0433, 0x0434, 0x0435, 0x0436, 0x0437, 0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E, 0x043F,
            wchar32('z'),
            0x0440, 0x0441, 0x0442, 0x0443, 0x0444, 0x0445, 0x0446, 0x0447, 0x0448, 0x0449, 0x044A, 0x044B, 0x044C, 0x044D, 0x044E, 0x044F,
            wchar32('z'),
            0x00};

        TUtf32String strWide(cyrilicAndLatinWide);
        TString strUtf8 = WideToUTF8(strWide);

        UNIT_ASSERT(strWide == TUtf32String::FromUtf8(strUtf8.c_str()));
        UNIT_ASSERT(strWide == TUtf32String::FromUtf8(strUtf8));
        UNIT_ASSERT(strWide == TUtf32String::FromUtf8(TStringBuf(strUtf8)));

        // assign

        TUtf32String s1;
        s1.AssignAscii("1234");
        UNIT_ASSERT(s1 == ASCIIToUTF32("1234"));

        s1.AssignUtf8(strUtf8);
        UNIT_ASSERT(s1 == strWide);

        s1.AssignAscii(text);
        UNIT_ASSERT(s1 == wtext);

        // append

        TUtf32String s2;
        TUtf32String testAppend = strWide;
        s2.AppendUtf16(UTF8ToWide(strUtf8));
        UNIT_ASSERT(testAppend == s2);

        testAppend += ' ';
        s2.AppendAscii(" ");
        UNIT_ASSERT(testAppend == s2);

        testAppend += '_';
        s2.AppendUtf8("_");
        UNIT_ASSERT(testAppend == s2);

        testAppend += wtext;
        s2.AppendAscii(text);
        UNIT_ASSERT(testAppend == s2);

        testAppend += wtext;
        s2.AppendUtf8(text);

        UNIT_ASSERT(testAppend == s2);
    }

    void TestLetOperator() {
        TUtf32String str;

        str = wchar32('X');
        UNIT_ASSERT(str == TUtf32String::FromAscii("X"));

        const TUtf32String hello = TUtf32String::FromAscii("hello");
        str = hello.data();
        UNIT_ASSERT(str == hello);

        str = hello;
        UNIT_ASSERT(str == hello);
    }

    void TestStringLiterals() {
        TUtf32String s1 = U"hello";
        UNIT_ASSERT_VALUES_EQUAL(s1, TUtf32String::FromAscii("hello"));

        TUtf32String s2 = U"привет";
        UNIT_ASSERT_VALUES_EQUAL(s2, TUtf32String::FromUtf8("привет"));
    }
};

UNIT_TEST_SUITE_REGISTRATION(TUtf32StringTest);

class TStringStdTest: public TTestBase, private TStringStdTestImpl<TString, TTestData<char>> {
public:
    UNIT_TEST_SUITE(TStringStdTest);
    UNIT_TEST(constructor);
    UNIT_TEST(reserve);
    UNIT_TEST(short_string);
    UNIT_TEST(erase);
    UNIT_TEST(data);
    UNIT_TEST(c_str);
    UNIT_TEST(null_char);
    UNIT_TEST(insert);
    UNIT_TEST(resize);
    UNIT_TEST(find);
    UNIT_TEST(capacity);
    UNIT_TEST(assign);
    UNIT_TEST(copy);
    UNIT_TEST(cbegin_cend);
    UNIT_TEST(compare);
    UNIT_TEST(find_last_of);
#if 0
        UNIT_TEST(rfind);
        UNIT_TEST(replace);
#endif
    UNIT_TEST(find_last_not_of);
    UNIT_TEST_SUITE_END();
};

UNIT_TEST_SUITE_REGISTRATION(TStringStdTest);

class TWideStringStdTest: public TTestBase, private TStringStdTestImpl<TUtf16String, TTestData<wchar16>> {
public:
    UNIT_TEST_SUITE(TWideStringStdTest);
    UNIT_TEST(constructor);
    UNIT_TEST(reserve);
    UNIT_TEST(short_string);
    UNIT_TEST(erase);
    UNIT_TEST(data);
    UNIT_TEST(c_str);
    UNIT_TEST(null_char);
    UNIT_TEST(insert);
    UNIT_TEST(resize);
    UNIT_TEST(find);
    UNIT_TEST(capacity);
    UNIT_TEST(assign);
    UNIT_TEST(copy);
    UNIT_TEST(cbegin_cend);
    UNIT_TEST(compare);
    UNIT_TEST(find_last_of);
#if 0
        UNIT_TEST(rfind);
        UNIT_TEST(replace);
#endif
    UNIT_TEST(find_last_not_of);
    UNIT_TEST_SUITE_END();
};

UNIT_TEST_SUITE_REGISTRATION(TWideStringStdTest);

Y_UNIT_TEST_SUITE(TStringConversionTest) {
    Y_UNIT_TEST(ConversionToStdStringTest) {
        TString abra = "cadabra";
        std::string stdAbra = abra;
        UNIT_ASSERT_VALUES_EQUAL(stdAbra, "cadabra");
    }

    Y_UNIT_TEST(ConversionToStdStringViewTest) {
        TString abra = "cadabra";
        std::string_view stdAbra = abra;
        UNIT_ASSERT_VALUES_EQUAL(stdAbra, "cadabra");
    }
}

Y_UNIT_TEST_SUITE(HashFunctorTests) {
    Y_UNIT_TEST(TestTransparency) {
        THash<TString> h;
        const char* ptr = "a";
        const TStringBuf strbuf = ptr;
        const TString str = ptr;
        const std::string stdStr = ptr;
        UNIT_ASSERT_VALUES_EQUAL(h(ptr), h(strbuf));
        UNIT_ASSERT_VALUES_EQUAL(h(ptr), h(str));
        UNIT_ASSERT_VALUES_EQUAL(h(ptr), h(stdStr));
    }
}
