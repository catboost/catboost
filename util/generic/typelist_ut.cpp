#include <library/unittest/registar.h>

#include "typelist.h"
#include "vector.h"
#include "map.h"

#include "type_name.h"

class TTypeListTest: public TTestBase {
    UNIT_TEST_SUITE(TTypeListTest);
    UNIT_TEST(TestSimple);
    UNIT_TEST(TestHave);
    UNIT_TEST(TestGet);
    UNIT_TEST(TestSelectBy);
    UNIT_TEST_SUITE_END();

public:
    void TestGet() {
        using TList = TTypeList<int, char, float>;

        UNIT_ASSERT_TYPES_EQUAL(TList::TGet<0>, int);
        UNIT_ASSERT_TYPES_EQUAL(TList::TGet<1>, char);
        UNIT_ASSERT_TYPES_EQUAL(TList::TGet<2>, float);
    }

    void TestSimple() {
        using TList = TTypeList<int, char, float>;
        UNIT_ASSERT_EQUAL(TList::Length, 3);
        UNIT_ASSERT_TYPES_EQUAL(TList::THead, int);
    }

    struct TA {};
    struct TB {};
    struct TC {};
    void TestHave() {
        using TList = TTypeList<TA, TB*, const TC&>;
        UNIT_ASSERT(TList::THave<TA>::Result);
        UNIT_ASSERT(TList::THave<TB*>::Result);
        UNIT_ASSERT(!TList::THave<TB>::Result);
        UNIT_ASSERT(TList::THave<const TC&>::Result);
        UNIT_ASSERT(!TList::THave<TC&>::Result);
    }

    template <class T>
    class TT {};

    template <class T>
    struct TIs1ArgTemplate {
        enum {
            Result = false
        };
    };

    template <class T, template <class> class TT>
    struct TIs1ArgTemplate<TT<T>> {
        enum {
            Result = true
        };
    };

    template <class T>
    struct TIsNArgTemplate {
        enum {
            Result = false
        };
    };

    template <template <class...> class TT, class... R>
    struct TIsNArgTemplate<TT<R...>> {
        enum {
            Result = true
        };
    };
    template <class>
    struct TAnyType {
        enum {
            Result = true
        };
    };

    template <class T>
    struct TMyVector {};

    template <class T1, class T2>
    struct TMyMap {};

    void TestSelectBy() {
        using TList = TTypeList<TA, TB, TMyMap<TA*, TB>, TMyVector<TA>, TC>;

        UNIT_ASSERT_TYPES_EQUAL(TList::TSelectBy<TAnyType>::TResult, TA);
        UNIT_ASSERT_TYPES_EQUAL(TList::TSelectBy<TIs1ArgTemplate>::TResult, TMyVector<TA>);
        using TMyMapPTATB = TMyMap<TA*, TB>;
        UNIT_ASSERT_TYPES_EQUAL(TList::TSelectBy<TIsNArgTemplate>::TResult, TMyMapPTATB);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TTypeListTest);
