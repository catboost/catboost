#include "va_args.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TMacroVarargMapTest) {
    Y_UNIT_TEST(TestMapArgs) {
        static const char COMBINED[] = Y_MAP_ARGS(Y_STRINGIZE, 1, 2, 3);
        UNIT_ASSERT_STRINGS_EQUAL(COMBINED, "123");
    }

    Y_UNIT_TEST(TestMapArgsWithLast) {
#define ADD(x) x +
#define ID(x) x
        static const int SUM = Y_MAP_ARGS_WITH_LAST(ADD, ID, 1, 2, 3, 4 + 5);
        UNIT_ASSERT_VALUES_EQUAL(SUM, 1 + 2 + 3 + 4 + 5);
#undef ADD
#undef ID
    }

    Y_UNIT_TEST(TestMapArgsN) {
#define MAP_ARG(INDEX, X) Y_STRINGIZE(X)
#define MAP_INDEX(INDEX, X) Y_STRINGIZE(INDEX)
        static const char COMBINED_ARGS[] = Y_MAP_ARGS_N(MAP_ARG, 1, 2, 3);
        UNIT_ASSERT_STRINGS_EQUAL(COMBINED_ARGS, "123");
        static const char COMBINED_INDEXES[] = Y_MAP_ARGS_N(MAP_INDEX, 1, 2, 3);
        UNIT_ASSERT_STRINGS_EQUAL(COMBINED_INDEXES, "321");
#undef MAP_INDEX
#undef MAP_ARG
    }

    Y_UNIT_TEST(TestMapArgsWithLastN) {
#define ADD_ARG(INDEX, X) X +
#define ID_ARG(INDEX, X) X
#define MAP_INDEX(INDEX, X) Y_STRINGIZE(INDEX)
        static const int SUM = Y_MAP_ARGS_WITH_LAST_N(ADD_ARG, ID_ARG, 1, 2, 3, 4 + 5);
        UNIT_ASSERT_VALUES_EQUAL(SUM, 1 + 2 + 3 + 4 + 5);
        static const char COMBINED_INDEXES[] = Y_MAP_ARGS_WITH_LAST_N(MAP_INDEX, MAP_INDEX, 1, 2, 3, 4 + 5);
        UNIT_ASSERT_STRINGS_EQUAL(COMBINED_INDEXES, "4321");
#undef MAP_INDEX
#undef ADD_ARG
#undef ID_ARG
    }
} // Y_UNIT_TEST_SUITE(TMacroVarargMapTest)

Y_UNIT_TEST_SUITE(TestVaArgs) {
    Y_UNIT_TEST(Count) {
        // UNIT_ASSERT((Y_COUNT_ARGS() == 0));  // FIXME: make this case work after __VA_OPT__ (c++20)
        UNIT_ASSERT((Y_COUNT_ARGS(1) == 1));
        UNIT_ASSERT((Y_COUNT_ARGS(1, 2) == 2));
        UNIT_ASSERT((Y_COUNT_ARGS(1, 2, 3) == 3));
        UNIT_ASSERT((Y_COUNT_ARGS(1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0) == 20));
    }

    Y_UNIT_TEST(GetElem) {
        UNIT_ASSERT((Y_GET_ARG(0, 1) == 1));
        UNIT_ASSERT((Y_GET_ARG(0, 0, 1, 2, 3, 4, 5) == 0));
        UNIT_ASSERT((Y_GET_ARG(1, 0, 1, 2, 3, 4, 5) == 1));
        UNIT_ASSERT((Y_GET_ARG(2, 0, 1, 2, 3, 4, 5) == 2));
        UNIT_ASSERT((Y_GET_ARG(3, 0, 1, 2, 3, 4, 5) == 3));
        UNIT_ASSERT((Y_GET_ARG(4, 0, 1, 2, 3, 4, 5) == 4));
        UNIT_ASSERT((Y_GET_ARG(5, 0, 1, 2, 3, 4, 5) == 5));
    }

    Y_UNIT_TEST(MapArgs) {
#define MAP(x) x + /* NOLINT */
        // UNIT_ASSERT((Y_MAP_ARGS(MAP) 0 == 0));  // FIXME: make this case work after __VA_OPT__ (c++20)
        UNIT_ASSERT((Y_MAP_ARGS(MAP, 1, 2, 3, 4) 0 == 10));
#undef MAP
    }

    Y_UNIT_TEST(MapArgsWithLast) {
#define MAP(x) x + /* NOLINT */
#define MAP_LAST(x) x
        UNIT_ASSERT((Y_MAP_ARGS_WITH_LAST(MAP, MAP_LAST, 1, 2, 3, 4) == 10));
#undef MAP_LAST
#undef MAP
    }

    Y_UNIT_TEST(AllButLast) {
        const char array[] = {Y_ALL_BUT_LAST(1, 2, 3, 4, 5)};
        UNIT_ASSERT((sizeof(array) == 4));
        UNIT_ASSERT((array[0] == 1));
        UNIT_ASSERT((array[1] == 2));
        UNIT_ASSERT((array[2] == 3));
        UNIT_ASSERT((array[3] == 4));
    }

    Y_UNIT_TEST(Last) {
        UNIT_ASSERT((Y_LAST(1) == 1));
        UNIT_ASSERT((Y_LAST(1, 2, 3) == 3));
    }

    Y_UNIT_TEST(ImplDispatcher) {
#define I1(x) (x)
#define I2(x, y) ((x) + (y))
#define I3(x, y, z) ((x) + (y) + (z))
#define I(...) Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_3(__VA_ARGS__, I3, I2, I1)(__VA_ARGS__))
        UNIT_ASSERT((I(1) == 1));
        UNIT_ASSERT((I(1, 2) == 3));
        UNIT_ASSERT((I(1, 2, 3) == 6));
#undef I
#undef I3
#undef I2
#undef I1
    }
} // Y_UNIT_TEST_SUITE(TestVaArgs)
