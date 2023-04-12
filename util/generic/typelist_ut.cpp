#include <library/cpp/testing/unittest/registar.h>
#include <util/system/type_name.h>

#include "typelist.h"
#include "vector.h"

class TTypeListTest: public TTestBase {
    UNIT_TEST_SUITE(TTypeListTest);
    UNIT_TEST(TestSimple);
    UNIT_TEST(TestHave);
    UNIT_TEST(TestGet);
    UNIT_TEST(TestFloatList);
    UNIT_TEST(TestSelectBy);
    UNIT_TEST_SUITE_END();

public:
    void TestGet() {
        using TListType = TTypeList<int, char, float>;

        UNIT_ASSERT_TYPES_EQUAL(TListType::TGet<0>, int);
        UNIT_ASSERT_TYPES_EQUAL(TListType::TGet<1>, char);
        UNIT_ASSERT_TYPES_EQUAL(TListType::TGet<2>, float);
    }

    void TestSimple() {
        using TListType = TTypeList<int, char, float>;
        UNIT_ASSERT_EQUAL(TListType::Length, 3);
        UNIT_ASSERT_TYPES_EQUAL(TListType::THead, int);
    }

    struct TA {};
    struct TB {};
    struct TC {};
    void TestHave() {
        using TListType = TTypeList<TA, TB*, const TC&>;
        UNIT_ASSERT(TListType::THave<TA>::value);
        UNIT_ASSERT(TListType::THave<TB*>::value);
        UNIT_ASSERT(!TListType::THave<TB>::value);
        UNIT_ASSERT(TListType::THave<const TC&>::value);
        UNIT_ASSERT(!TListType::THave<TC&>::value);
    }

    template <class T>
    class TT {};

    template <class T>
    struct TIs1ArgTemplate: std::false_type {};

    template <class T, template <class> class TT>
    struct TIs1ArgTemplate<TT<T>>: std::true_type {};

    template <class T>
    struct TIsNArgTemplate: std::false_type {};

    template <template <class...> class TT, class... R>
    struct TIsNArgTemplate<TT<R...>>: std::true_type {};

    template <class>
    struct TAnyType: std::true_type {};

    template <class T>
    struct TMyVector {};

    template <class T1, class T2>
    struct TMyMap {};

    void TestSelectBy() {
        using TListType = TTypeList<TA, TB, TMyMap<TA*, TB>, TMyVector<TA>, TC>;

        UNIT_ASSERT_TYPES_EQUAL(TListType::TSelectBy<TAnyType>::type, TA);
        UNIT_ASSERT_TYPES_EQUAL(TListType::TSelectBy<TIs1ArgTemplate>::type, TMyVector<TA>);
        using TMyMapPTATB = TMyMap<TA*, TB>;
        UNIT_ASSERT_TYPES_EQUAL(TListType::TSelectBy<TIsNArgTemplate>::type, TMyMapPTATB);
    }

    void TestFloatList() {
        UNIT_ASSERT_TYPES_EQUAL(TFixedWidthFloat<ui32>, float);
        UNIT_ASSERT_TYPES_EQUAL(TFixedWidthFloat<i32>, float);
        UNIT_ASSERT_TYPES_EQUAL(TFixedWidthFloat<ui64>, double);
        UNIT_ASSERT_TYPES_EQUAL(TFixedWidthFloat<i64>, double);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TTypeListTest);
