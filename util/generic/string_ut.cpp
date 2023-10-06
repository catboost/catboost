#include "deque.h"
#include "strbuf.h"
#include "string_ut.h"
#include "vector.h"
#include "yexception.h"

#include <util/charset/wide.h>
#include <util/str_stl.h>
#include <util/stream/output.h>
#include <util/string/subst.h>

#include <string>
#include <sstream>
#include <algorithm>
#include <stdexcept>

#ifdef TSTRING_IS_STD_STRING
static_assert(sizeof(TString) == sizeof(std::string), "expect sizeof(TString) == sizeof(std::string)");
#else
static_assert(sizeof(TString) == sizeof(const char*), "expect sizeof(TString) == sizeof(const char*)");
#endif

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
    using TChar = typename TStringType::char_type;
    using TTraits = typename TStringType::traits_type;
    using TView = std::basic_string_view<TChar, TTraits>;

    TTestData Data_;

protected:
    void Constructor() {
        UNIT_ASSERT_EXCEPTION(TStringType((size_t)-1, *Data_.a()), std::length_error);
    }

    void reserve() {
#if 0
        TStringType s;
        UNIT_ASSERT_EXCEPTION(s.reserve(s.max_size() + 1), std::length_error);

        // Non-shared behaviour - never shrink

        s.reserve(256);
    #ifndef TSTRING_IS_STD_STRING
        const auto* data = s.data();

        UNIT_ASSERT(s.capacity() >= 256);

        s.reserve(128);

        UNIT_ASSERT(s.capacity() >= 256 && s.data() == data);
    #endif

        s.resize(64, 'x');
        s.reserve(10);

    #ifdef TSTRING_IS_STD_STRING
        UNIT_ASSERT(s.capacity() >= 64);
    #else
        UNIT_ASSERT(s.capacity() >= 256 && s.data() == data);
    #endif

    #ifndef TSTRING_IS_STD_STRING
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
    #endif
#endif
    }

    void short_string() {
        TStringType const ref_short_str1(Data_.str1()), ref_short_str2(Data_.str2());
        TStringType short_str1(ref_short_str1), short_str2(ref_short_str2);
        TStringType const ref_long_str1(Data_.str__________________________________________________1());
        TStringType const ref_long_str2(Data_.str__________________________________________________2());
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

            UNIT_ASSERT(str_vect[0] == ref_short_str1);
            UNIT_ASSERT(str_vect[1] == ref_long_str1);
            UNIT_ASSERT(str_vect[2] == ref_short_str2);
            UNIT_ASSERT(str_vect[3] == ref_long_str2);
        }
    }

    void erase() {
        TChar const* c_str = Data_.Hello_World();
        TStringType str(c_str);
        UNIT_ASSERT(str == c_str);

        str.erase(str.begin() + 1, str.end() - 1); // Erase all but first and last.

        size_t i;
        for (i = 0; i < str.size(); ++i) {
            switch (i) {
                case 0:
                    UNIT_ASSERT(str[i] == *Data_.H());
                    break;

                case 1:
                    UNIT_ASSERT(str[i] == *Data_.d());
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
                    UNIT_ASSERT(str[i] == *Data_.H());
                    break;

                case 1:
                    UNIT_ASSERT(str[i] == *Data_.d());
                    break;

                default:
                    UNIT_ASSERT(false);
            }
        }

        str.erase(1);
        UNIT_ASSERT(str == Data_.H());
    }

    void data() {
        TStringType xx;

        // ISO-IEC-14882:1998(E), 21.3.6, paragraph 3
        UNIT_ASSERT(xx.data() != nullptr);
    }

    void c_str() {
        TStringType low(Data_._2004_01_01());
        TStringType xx;
        TStringType yy;

        // ISO-IEC-14882:1998(E), 21.3.6, paragraph 1
        UNIT_ASSERT(*(yy.c_str()) == 0);

        // Blocks A and B should follow each other.
        // Block A:
        xx = Data_._123456();
        xx += low;
        UNIT_ASSERT(xx.c_str() == TView(Data_._1234562004_01_01()));
        // End of block A

        // Block B:
        xx = Data_._1234();
        xx += Data_._5();
        UNIT_ASSERT(xx.c_str() == TView(Data_._12345()));
        // End of block B
    }

    void null_char_of_empty() {
        const TStringType s;

        //NOTE: https://a.yandex-team.ru/arcadia/junk/grechnik/test_string?rev=r12602052
        i64 i = s[s.size()];
        UNIT_ASSERT_VALUES_EQUAL(i, 0);
    }

    void null_char() {
        // ISO/IEC 14882:1998(E), ISO/IEC 14882:2003(E), 21.3.4 ('... the const version')
        const TStringType s(Data_._123456());

        UNIT_ASSERT(s[s.size()] == 0);
    }

    // Allowed since C++17, see http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2475
    void null_char_assignment_to_subscript_of_empty() {
        TStringType s;

#ifdef TSTRING_IS_STD_STRING
        using reference = volatile typename TStringType::value_type&;
#else
        using reference = typename TStringType::reference;
#endif
        reference trailing_zero = s[s.size()];
        trailing_zero = 0;
        UNIT_ASSERT(trailing_zero == 0);
    }

    // Allowed since C++17, see http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2475
    void null_char_assignment_to_subscript_of_nonempty() {
        TStringType s(Data_._123456());

#ifdef TSTRING_IS_STD_STRING
        using reference = volatile typename TStringType::value_type&;
#else
        using reference = typename TStringType::reference;
#endif
        reference trailing_zero = s[s.size()];
        trailing_zero = 0;
        UNIT_ASSERT(trailing_zero == 0);
    }

#ifndef TSTRING_IS_STD_STRING
    // Dereferencing string end() is not allowed by C++ standard as of C++20, avoid using in real code.
    void null_char_assignment_to_end_of_empty() {
        TStringType s;

        volatile auto& trailing_zero = *(s.begin() + s.size());
        trailing_zero = 0;
        UNIT_ASSERT(trailing_zero == 0);
    }

    // Dereferencing string end() is not allowed by C++ standard as of C++20, avoid using in real code.
    void null_char_assignment_to_end_of_nonempty() {
        TStringType s(Data_._123456());

        volatile auto& trailing_zero = *(s.begin() + s.size());
        trailing_zero = 0;
        UNIT_ASSERT(trailing_zero == 0);
    }
#endif

    void insert() {
        TStringType strorg = Data_.This_is_test_string_for_string_calls();
        TStringType str;

        // In case of reallocation there is no auto reference problem
        // so we reserve a big enough TStringType to be sure to test this
        // particular point.

        str.reserve(100);
        str = strorg;

        //test self insertion:
        str.insert(10, str.c_str() + 5, 15);
        UNIT_ASSERT(str == Data_.This_is_teis_test_string_st_string_for_string_calls());

        str = strorg;
        str.insert(15, str.c_str() + 5, 25);
        UNIT_ASSERT(str == Data_.This_is_test_stis_test_string_for_stringring_for_string_calls());

        str = strorg;
        str.insert(0, str.c_str() + str.size() - 4, 4);
        UNIT_ASSERT(str == Data_.allsThis_is_test_string_for_string_calls());

        str = strorg;
        str.insert(0, str.c_str() + str.size() / 2 - 1, str.size() / 2 + 1);
        UNIT_ASSERT(str == Data_.ng_for_string_callsThis_is_test_string_for_string_calls());

        str = strorg;
        typename TStringType::iterator b = str.begin();
        typename TStringType::const_iterator s = str.begin() + str.size() / 2 - 1;
        typename TStringType::const_iterator e = str.end();
        str.insert(b, s, e);
        UNIT_ASSERT(str == Data_.ng_for_string_callsThis_is_test_string_for_string_calls());

#if 0
        // AV
        str = strorg;
        str.insert(str.begin(), str.begin() + str.size() / 2 - 1, str.end());
        UNIT_ASSERT(str == Data.ng_for_string_callsThis_is_test_string_for_string_calls());
#endif

        TStringType str0;
        str0.insert(str0.begin(), 5, *Data_._0());
        UNIT_ASSERT(str0 == Data_._00000());

        TStringType str1;
        {
            typename TStringType::size_type pos = 0, nb = 2;
            str1.insert(pos, nb, *Data_._1());
        }
        UNIT_ASSERT(str1 == Data_._11());

        str0.insert(0, str1);
        UNIT_ASSERT(str0 == Data_._1100000());

        TStringType str2(Data_._2345());
        str0.insert(str0.size(), str2, 1, 2);
        UNIT_ASSERT(str0 == Data_._110000034());

        str1.insert(str1.begin() + 1, 2, *Data_._2());
        UNIT_ASSERT(str1 == Data_._1221());

        str1.insert(2, Data_._333333(), 3);
        UNIT_ASSERT(str1 == Data_._1233321());

        str1.insert(4, Data_._4444());
        UNIT_ASSERT(str1 == Data_._12334444321());

        str1.insert(str1.begin() + 6, *Data_._5());
        UNIT_ASSERT(str1 == Data_._123344544321());
    }

    void resize() {
        TStringType s;

        s.resize(0);

        UNIT_ASSERT(*s.c_str() == 0);

        s = Data_._1234567();

        s.resize(0);
        UNIT_ASSERT(*s.c_str() == 0);

        s = Data_._1234567();
        s.resize(1);
        UNIT_ASSERT(s.size() == 1);
        UNIT_ASSERT(*s.c_str() == *Data_._1());
        UNIT_ASSERT(*(s.c_str() + 1) == 0);

        s = Data_._1234567();
#if 0
        s.resize(10);
#else
        s.resize(10, 0);
#endif
        UNIT_ASSERT(s.size() == 10);
        UNIT_ASSERT(s[6] == *Data_._7());
        UNIT_ASSERT(s[7] == 0);
        UNIT_ASSERT(s[8] == 0);
        UNIT_ASSERT(s[9] == 0);
    }

    void find() {
        TStringType s(Data_.one_two_three_one_two_three());

        UNIT_ASSERT(s.find(Data_.one()) == 0);
        UNIT_ASSERT(s.find(*Data_.t()) == 4);
        UNIT_ASSERT(s.find(*Data_.t(), 5) == 8);

        UNIT_ASSERT(s.find(Data_.four()) == TStringType::npos);
        UNIT_ASSERT(s.find(Data_.one(), TStringType::npos) == TStringType::npos);
        UNIT_ASSERT(s.find_first_of(Data_.abcde()) == 2);
        UNIT_ASSERT(s.find_first_not_of(Data_.enotw_()) == 9);
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
        TChar const* cstr = Data_.test_string_for_assign();

        s.assign(cstr, cstr + 22);
        UNIT_ASSERT(s == Data_.test_string_for_assign());

        TStringType s2(Data_.other_test_string());
        s.assign(s2);
        UNIT_ASSERT(s == s2);

        static TStringType str1;
        static TStringType str2;

        // short TStringType optim:
        str1 = Data_._123456();
        // longer than short TStringType:
        str2 = Data_._1234567890123456789012345678901234567890();

        UNIT_ASSERT(str1[5] == *Data_._6());
        UNIT_ASSERT(str2[29] == *Data_._0());
    }

    void copy() {
        TStringType s(Data_.foo());
        TChar dest[4];
        dest[0] = dest[1] = dest[2] = dest[3] = 1;
        s.copy(dest, 4);
        int pos = 0;
        UNIT_ASSERT(dest[pos++] == *Data_.f());
        UNIT_ASSERT(dest[pos++] == *Data_.o());
        UNIT_ASSERT(dest[pos++] == *Data_.o());
        UNIT_ASSERT(dest[pos++] == 1);

        dest[0] = dest[1] = dest[2] = dest[3] = 1;
        s.copy(dest, 4, 2);
        pos = 0;
        UNIT_ASSERT(dest[pos++] == *Data_.o());
        UNIT_ASSERT(dest[pos++] == 1);

        UNIT_ASSERT_EXCEPTION(s.copy(dest, 4, 5), std::out_of_range);
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
        TStringType str1(Data_.abcdef());
        TStringType str2;

        str2 = Data_.abcdef();
        UNIT_ASSERT(str1.compare(str2) == 0);
        UNIT_ASSERT(str1.compare(str2.data(), str2.size()) == 0);
        str2 = Data_.abcde();
        UNIT_ASSERT(str1.compare(str2) > 0);
        UNIT_ASSERT(str1.compare(str2.data(), str2.size()) > 0);
        str2 = Data_.abcdefg();
        UNIT_ASSERT(str1.compare(str2) < 0);
        UNIT_ASSERT(str1.compare(str2.data(), str2.size()) < 0);

        UNIT_ASSERT(str1.compare(Data_.abcdef()) == 0);
        UNIT_ASSERT(str1.compare(Data_.abcde()) > 0);
        UNIT_ASSERT(str1.compare(Data_.abcdefg()) < 0);

        str2 = Data_.cde();
        UNIT_ASSERT(str1.compare(2, 3, str2) == 0);
        str2 = Data_.cd();
        UNIT_ASSERT(str1.compare(2, 3, str2) > 0);
        str2 = Data_.cdef();
        UNIT_ASSERT(str1.compare(2, 3, str2) < 0);

        str2 = Data_.abcdef();
        UNIT_ASSERT(str1.compare(2, 3, str2, 2, 3) == 0);
        UNIT_ASSERT(str1.compare(2, 3, str2, 2, 2) > 0);
        UNIT_ASSERT(str1.compare(2, 3, str2, 2, 4) < 0);

        UNIT_ASSERT(str1.compare(2, 3, Data_.cdefgh(), 3) == 0);
        UNIT_ASSERT(str1.compare(2, 3, Data_.cdefgh(), 2) > 0);
        UNIT_ASSERT(str1.compare(2, 3, Data_.cdefgh(), 4) < 0);
    }

    void find_last_of() {
        // 21.3.6.4
        TStringType s(Data_.one_two_three_one_two_three());

        UNIT_ASSERT(s.find_last_of(Data_.abcde()) == 26);
        UNIT_ASSERT(s.find_last_of(TStringType(Data_.abcde())) == 26);

        TStringType test(Data_.aba());

        UNIT_ASSERT(test.find_last_of(Data_.a(), 2, 1) == 2);
        UNIT_ASSERT(test.find_last_of(Data_.a(), 1, 1) == 0);
        UNIT_ASSERT(test.find_last_of(Data_.a(), 0, 1) == 0);

        UNIT_ASSERT(test.find_last_of(*Data_.a(), 2) == 2);
        UNIT_ASSERT(test.find_last_of(*Data_.a(), 1) == 0);
        UNIT_ASSERT(test.find_last_of(*Data_.a(), 0) == 0);
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
        TStringType s(Data_.one_two_three_one_two_three());

        UNIT_ASSERT(s.find_last_not_of(Data_.ehortw_()) == 15);

        TStringType test(Data_.aba());

        UNIT_ASSERT(test.find_last_not_of(Data_.a(), 2, 1) == 1);
        UNIT_ASSERT(test.find_last_not_of(Data_.b(), 2, 1) == 2);
        UNIT_ASSERT(test.find_last_not_of(Data_.a(), 1, 1) == 1);
        UNIT_ASSERT(test.find_last_not_of(Data_.b(), 1, 1) == 0);
        UNIT_ASSERT(test.find_last_not_of(Data_.a(), 0, 1) == TStringType::npos);
        UNIT_ASSERT(test.find_last_not_of(Data_.b(), 0, 1) == 0);

        UNIT_ASSERT(test.find_last_not_of(*Data_.a(), 2) == 1);
        UNIT_ASSERT(test.find_last_not_of(*Data_.b(), 2) == 2);
        UNIT_ASSERT(test.find_last_not_of(*Data_.a(), 1) == 1);
        UNIT_ASSERT(test.find_last_not_of(*Data_.b(), 1) == 0);
        UNIT_ASSERT(test.find_last_not_of(*Data_.a(), 0) == TStringType::npos);
        UNIT_ASSERT(test.find_last_not_of(*Data_.b(), 0) == 0);
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
        deque<TChar> cdeque;
        cdeque.push_back(*Data.I());
        str.replace(str.begin(), str.begin() + 11, cdeque.begin(), cdeque.end());
        UNIT_ASSERT(str == Data.Is_test_StringT_for_StringT_calls());
    #endif
    }
#endif
}; // TStringStdTestImpl

class TStringTest: public TTestBase, private TStringTestImpl<TString, TTestData<char>> {
public:
    UNIT_TEST_SUITE(TStringTest);
    UNIT_TEST(TestMaxSize);
    UNIT_TEST(TestConstructors);
    UNIT_TEST(TestReplace);
#ifndef TSTRING_IS_STD_STRING
    UNIT_TEST(TestRefCount);
#endif
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
#ifndef TSTRING_IS_STD_STRING
    UNIT_TEST(TestCharRef);
#endif
    UNIT_TEST(TestBack)
    UNIT_TEST(TestFront)
    UNIT_TEST(TestIterators);
    UNIT_TEST(TestReverseIterators);
    UNIT_TEST(TestAppendUtf16)
    UNIT_TEST(TestFillingAssign)
    UNIT_TEST(TestStdStreamApi)
    //UNIT_TEST(TestOperatorsCI); must fail
    UNIT_TEST_SUITE_END();

    void TestAppendUtf16() {
        TString appended = TString("А роза упала").AppendUtf16(u" на лапу Азора");
        UNIT_ASSERT(appended == "А роза упала на лапу Азора");
    }

    void TestFillingAssign() {
        TString s("abc");
        s.assign(5, 'a');
        UNIT_ASSERT_VALUES_EQUAL(s, "aaaaa");
    }

    void TestStdStreamApi() {
        const TString data = "abracadabra";
        std::stringstream ss;
        ss << data;

        UNIT_ASSERT_VALUES_EQUAL(data, ss.str());

        ss << '\n'
           << data << std::endl;

        TString read = "xxx";
        ss >> read;
        UNIT_ASSERT_VALUES_EQUAL(read, data);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TStringTest);

class TWideStringTest: public TTestBase, private TStringTestImpl<TUtf16String, TTestData<wchar16>> {
public:
    UNIT_TEST_SUITE(TWideStringTest);
    UNIT_TEST(TestConstructors);
    UNIT_TEST(TestReplace);
#ifndef TSTRING_IS_STD_STRING
    UNIT_TEST(TestRefCount);
#endif
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
#ifndef TSTRING_IS_STD_STRING
    UNIT_TEST(TestCharRef);
#endif
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
#ifndef TSTRING_IS_STD_STRING
    UNIT_TEST(TestRefCount);
#endif
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
#ifndef TSTRING_IS_STD_STRING
    UNIT_TEST(TestCharRef);
#endif
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
        UNIT_ASSERT(strWide == UTF8ToUTF32<true>(strUtf8));
        UNIT_ASSERT(strWide == UTF8ToUTF32<false>(strUtf8));
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
    UNIT_TEST(Constructor);
    UNIT_TEST(reserve);
    UNIT_TEST(short_string);
    UNIT_TEST(erase);
    UNIT_TEST(data);
    UNIT_TEST(c_str);
    UNIT_TEST(null_char_of_empty);
    UNIT_TEST(null_char);
    UNIT_TEST(null_char_assignment_to_subscript_of_empty);
    UNIT_TEST(null_char_assignment_to_subscript_of_nonempty);
#ifndef TSTRING_IS_STD_STRING
    UNIT_TEST(null_char_assignment_to_end_of_empty);
    UNIT_TEST(null_char_assignment_to_end_of_nonempty);
#endif
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
    UNIT_TEST(Constructor);
    UNIT_TEST(reserve);
    UNIT_TEST(short_string);
    UNIT_TEST(erase);
    UNIT_TEST(data);
    UNIT_TEST(c_str);
    UNIT_TEST(null_char_of_empty);
    UNIT_TEST(null_char);
    UNIT_TEST(null_char_assignment_to_subscript_of_empty);
    UNIT_TEST(null_char_assignment_to_subscript_of_nonempty);
#ifndef TSTRING_IS_STD_STRING
    UNIT_TEST(null_char_assignment_to_end_of_empty);
    UNIT_TEST(null_char_assignment_to_end_of_nonempty);
#endif
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

#if !defined(TSTRING_IS_STD_STRING)
Y_UNIT_TEST_SUITE(StdNonConformant) {
    Y_UNIT_TEST(TestEraseNoThrow) {
        TString x;

        LegacyErase(x, 10);
    }

    Y_UNIT_TEST(TestReplaceNoThrow) {
        TString x;

        LegacyReplace(x, 0, 0, "1");

        UNIT_ASSERT_VALUES_EQUAL(x, "1");

        LegacyReplace(x, 10, 0, "1");

        UNIT_ASSERT_VALUES_EQUAL(x, "1");
    }

    Y_UNIT_TEST(TestNoAlias) {
        TString s = "x";

        s.AppendNoAlias("abc", 3);

        UNIT_ASSERT_VALUES_EQUAL(s, "xabc");
        UNIT_ASSERT_VALUES_EQUAL(TString(s.c_str()), "xabc");
    }
}
#endif

Y_UNIT_TEST_SUITE(Interop) {
    static void Mutate(std::string& s) {
        s += "y";
    }

    static void Mutate(TString& s) {
        Mutate(MutRef(s));
    }

    Y_UNIT_TEST(TestMutate) {
        TString x = "x";

        Mutate(x);

        UNIT_ASSERT_VALUES_EQUAL(x, "xy");
    }

    static std::string TransformStd(const std::string& s) {
        return s + "y";
    }

    static TString Transform(const TString& s) {
        return TransformStd(s);
    }

    Y_UNIT_TEST(TestTransform) {
        UNIT_ASSERT_VALUES_EQUAL(Transform(TString("x")), "xy");
    }

    Y_UNIT_TEST(TestTemp) {
        UNIT_ASSERT_VALUES_EQUAL("x" + ConstRef(TString("y")), "xy");
    }
}
