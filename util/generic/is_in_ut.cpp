#include <library/unittest/registar.h>

#include "algorithm.h"
#include "hash.h"
#include "hash_set.h"
#include "is_in.h"
#include "map.h"
#include "set.h"
#include "strbuf.h"
#include "string.h"

SIMPLE_UNIT_TEST_SUITE(TIsIn) {
    template <class TCont, class T>
    void TestIsInWithCont(const T& elem) {
        class TMapMock: public TCont {
        public:
            typename TCont::const_iterator find(const typename TCont::key_type& k) const {
                ++FindCalled;
                return TCont::find(k);
            }

            typename TCont::iterator find(const typename TCont::key_type& k) {
                ++FindCalled;
                return TCont::find(k);
            }

            mutable size_t FindCalled = 1;
        };

        TMapMock m;
        m.insert(elem);

        // use more effective find method
        UNIT_ASSERT(IsIn(m, "found"));
        UNIT_ASSERT(m.FindCalled);
        m.FindCalled = 0;

        UNIT_ASSERT(!IsIn(m, "not found"));
        UNIT_ASSERT(m.FindCalled);
        m.FindCalled = 0;
    }

    SIMPLE_UNIT_TEST(IsInTest) {
        TestIsInWithCont<ymap<TString, TString>>(std::make_pair("found", "1"));
        TestIsInWithCont<ymultimap<TString, TString>>(std::make_pair("found", "1"));
        TestIsInWithCont<yhash<TString, TString>>(std::make_pair("found", "1"));
        TestIsInWithCont<yhash_mm<TString, TString>>(std::make_pair("found", "1"));

        TestIsInWithCont<yset<TString>>("found");
        TestIsInWithCont<ymultiset<TString>>("found");
        TestIsInWithCont<yhash_set<TString>>("found");
        TestIsInWithCont<yhash_multiset<TString>>("found");

        // vector also compiles and works
        yvector<TString> v;
        v.push_back("found");
        UNIT_ASSERT(IsIn(v, "found"));
        UNIT_ASSERT(!IsIn(v, "not found"));

        // iterators interface
        UNIT_ASSERT(IsIn(v.begin(), v.end(), "found"));
        UNIT_ASSERT(!IsIn(v.begin(), v.end(), "not found"));

        // Works with TString (it has find, but find is not used)
        TString s = "found";
        UNIT_ASSERT(IsIn(s, 'f'));
        UNIT_ASSERT(!IsIn(s, 'z'));

        TStringBuf b = "found";
        UNIT_ASSERT(IsIn(b, 'f'));
        UNIT_ASSERT(!IsIn(b, 'z'));
    }

    SIMPLE_UNIT_TEST(IsInInitListTest) {
        const char* abc = "abc";
        const char* def = "def";

        UNIT_ASSERT(IsIn({6, 2, 12}, 6));
        UNIT_ASSERT(IsIn({6, 2, 12}, 2));
        UNIT_ASSERT(!IsIn({6, 2, 12}, 7));
        UNIT_ASSERT(IsIn({6}, 6));
        UNIT_ASSERT(!IsIn({6}, 7));
        UNIT_ASSERT(!IsIn(std::initializer_list<int>(), 6));
        UNIT_ASSERT(IsIn({STRINGBUF("abc"), STRINGBUF("def")}, STRINGBUF("abc")));
        UNIT_ASSERT(IsIn({STRINGBUF("abc"), STRINGBUF("def")}, STRINGBUF("def")));
        UNIT_ASSERT(IsIn({"abc", "def"}, STRINGBUF("def")));
        UNIT_ASSERT(IsIn({abc, def}, def)); // direct pointer comparison
        UNIT_ASSERT(!IsIn({STRINGBUF("abc"), STRINGBUF("def")}, STRINGBUF("ghi")));
        UNIT_ASSERT(!IsIn({"abc", "def"}, STRINGBUF("ghi")));
        UNIT_ASSERT(!IsIn({"abc", "def"}, TString("ghi")));

        const TStringBuf str = "abc////";

        UNIT_ASSERT(IsIn({"abc", "def"}, TStringBuf{~str, 3}));
    }

    SIMPLE_UNIT_TEST(ConfOfTest) {
        UNIT_ASSERT(IsIn({1, 2, 3}, 1));
        UNIT_ASSERT(!IsIn({1, 2, 3}, 4));

        const TString b = "b";

        UNIT_ASSERT(!IsIn({"a", "b", "c"}, ~b)); // compares pointers by value. Whether it's good or not.
        UNIT_ASSERT(IsIn(yvector<TStringBuf>({"a", "b", "c"}), ~b));
        UNIT_ASSERT(IsIn(yvector<TStringBuf>({"a", "b", "c"}), "b"));
    }
}
