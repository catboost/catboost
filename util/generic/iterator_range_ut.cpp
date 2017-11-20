#include "iterator_range.h"

#include <library/unittest/registar.h>
#include <util/generic/algorithm.h>
#include <util/generic/vector.h>

SIMPLE_UNIT_TEST_SUITE(IteratorRange) {
    SIMPLE_UNIT_TEST(DefaultConstructor) {
        TIteratorRange<int*> range;
        UNIT_ASSERT(range.empty());
    }

    SIMPLE_UNIT_TEST(RangeBasedForLoop) {
        // compileability test
        for (int i : TIteratorRange<int*>()) {
            Y_UNUSED(i);
        }
    }

    SIMPLE_UNIT_TEST(Works) {
        const int values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        auto range = MakeIteratorRange(values, values + Y_ARRAY_SIZE(values));
        UNIT_ASSERT_VALUES_EQUAL(range.size(), Y_ARRAY_SIZE(values));
        UNIT_ASSERT(Equal(range.begin(), range.end(), values));
        UNIT_ASSERT(!range.empty());
    }

    SIMPLE_UNIT_TEST(OperatorsAndReferences) {
        TVector<size_t> values{1, 2, 3, 4, 5};
        auto range = MakeIteratorRange(values.begin(), values.end());
        UNIT_ASSERT_VALUES_EQUAL(range[2], 3);
        UNIT_ASSERT_VALUES_EQUAL(range[range[2]], 4);
        *values.begin() = 100500;
        UNIT_ASSERT_VALUES_EQUAL(range[0], 100500);
        range[0] = 100501;
        UNIT_ASSERT_VALUES_EQUAL(range[0], 100501);

        TVector<bool> valuesBool{false, true, false, false, false, false, true};
        auto rangeBVector = MakeIteratorRange(valuesBool.begin(), valuesBool.end());
        UNIT_ASSERT_VALUES_EQUAL(rangeBVector[1], true);
        rangeBVector[0] = true;
        valuesBool.back() = false;
        UNIT_ASSERT_VALUES_EQUAL(rangeBVector[0], true);
        UNIT_ASSERT_VALUES_EQUAL(rangeBVector[2], false);
        UNIT_ASSERT_VALUES_EQUAL(rangeBVector[6], false);
    }

    SIMPLE_UNIT_TEST(CanUseInAlgorithms) {
        const int values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        auto range = MakeIteratorRange(values, values + Y_ARRAY_SIZE(values));
        // more like compileability test
        // we should be able to use TIteratorRange as a container parameter for standard algorithms
        UNIT_ASSERT(AllOf(range, [](int x) { return x > 0; }));
    }
}
