#include <library/cpp/testing/unittest/registar.h>

#include "algorithm.h"
#include "hash.h"
#include "hash_multi_map.h"
#include "hash_set.h"
#include "is_in.h"
#include "map.h"
#include "set.h"
#include "strbuf.h"
#include "string.h"

Y_UNIT_TEST_SUITE(TIsIn) {
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

    Y_UNIT_TEST(IsInTest) {
        TestIsInWithCont<TMap<TString, TString>>(std::make_pair("found", "1"));
        TestIsInWithCont<TMultiMap<TString, TString>>(std::make_pair("found", "1"));
        TestIsInWithCont<THashMap<TString, TString>>(std::make_pair("found", "1"));
        TestIsInWithCont<THashMultiMap<TString, TString>>(std::make_pair("found", "1"));

        TestIsInWithCont<TSet<TString>>("found");
        TestIsInWithCont<TMultiSet<TString>>("found");
        TestIsInWithCont<THashSet<TString>>("found");
        TestIsInWithCont<THashMultiSet<TString>>("found");

        // vector also compiles and works
        TVector<TString> v;
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

    Y_UNIT_TEST(IsInInitListTest) {
        const char* abc = "abc";
        const char* def = "def";

        UNIT_ASSERT(IsIn({6, 2, 12}, 6));
        UNIT_ASSERT(IsIn({6, 2, 12}, 2));
        UNIT_ASSERT(!IsIn({6, 2, 12}, 7));
        UNIT_ASSERT(IsIn({6}, 6));
        UNIT_ASSERT(!IsIn({6}, 7));
        UNIT_ASSERT(!IsIn(std::initializer_list<int>(), 6));
        UNIT_ASSERT(IsIn({TStringBuf("abc"), TStringBuf("def")}, TStringBuf("abc")));
        UNIT_ASSERT(IsIn({TStringBuf("abc"), TStringBuf("def")}, TStringBuf("def")));
        UNIT_ASSERT(IsIn({"abc", "def"}, TStringBuf("def")));
        UNIT_ASSERT(IsIn({abc, def}, def)); // direct pointer comparison
        UNIT_ASSERT(!IsIn({TStringBuf("abc"), TStringBuf("def")}, TStringBuf("ghi")));
        UNIT_ASSERT(!IsIn({"abc", "def"}, TStringBuf("ghi")));
        UNIT_ASSERT(!IsIn({"abc", "def"}, TString("ghi")));

        const TStringBuf str = "abc////";

        UNIT_ASSERT(IsIn({"abc", "def"}, TStringBuf{str.data(), 3}));
    }

    Y_UNIT_TEST(ConfOfTest) {
        UNIT_ASSERT(IsIn({1, 2, 3}, 1));
        UNIT_ASSERT(!IsIn({1, 2, 3}, 4));

        const TString b = "b";

        UNIT_ASSERT(!IsIn({"a", "b", "c"}, b.data())); // compares pointers by value. Whether it's good or not.
        UNIT_ASSERT(IsIn(TVector<TStringBuf>({"a", "b", "c"}), b.data()));
        UNIT_ASSERT(IsIn(TVector<TStringBuf>({"a", "b", "c"}), "b"));
    }

    Y_UNIT_TEST(IsInArrayTest) {
        const TString array[] = {"a", "b", "d"};

        UNIT_ASSERT(IsIn(array, "a"));
        UNIT_ASSERT(IsIn(array, TString("b")));
        UNIT_ASSERT(!IsIn(array, "c"));
        UNIT_ASSERT(IsIn(array, TStringBuf("d")));
    }
} // Y_UNIT_TEST_SUITE(TIsIn)
