#include "stack_vec.h"

#include <library/cpp/unittest/registar.h>

namespace {
    struct TNotCopyAssignable {
        const int Value;
    };

    static_assert(std::is_copy_constructible_v<TNotCopyAssignable>);
    static_assert(!std::is_copy_assignable_v<TNotCopyAssignable>);
}

Y_UNIT_TEST_SUITE(TStackBasedVectorTest) {
    Y_UNIT_TEST(TestCreateEmpty) {
        TStackVec<int> ints;
        UNIT_ASSERT_EQUAL(ints.size(), 0);
    }

    Y_UNIT_TEST(TestCreateNonEmpty) {
        TStackVec<int> ints(5);
        UNIT_ASSERT_EQUAL(ints.size(), 5);

        for (size_t i = 0; i < ints.size(); ++i) {
            UNIT_ASSERT_EQUAL(ints[i], 0);
        }
    }

    Y_UNIT_TEST(TestReallyOnStack) {
        TStackVec<int> ints(5);
        // Depends on libc++ std::vector layout, which is now __begin__, then __end__,
        // then __end_cap_ which is a __compressed_pair<pointer, allocator_type>
        UNIT_ASSERT_EQUAL((const char*)ints.data(), ((const char*)&ints) + 3 * sizeof(TStackVec<int>::pointer));
    }

    Y_UNIT_TEST(TestFallback) {
        TSmallVec<int> ints;
        for (int i = 0; i < 14; ++i) {
            ints.push_back(i);
        }

        for (size_t i = 0; i < ints.size(); ++i) {
            UNIT_ASSERT_EQUAL(ints[i], (int)i);
        }

        for (int i = 14; i < 20; ++i) {
            ints.push_back(i);
        }

        for (size_t i = 0; i < ints.size(); ++i) {
            UNIT_ASSERT_EQUAL(ints[i], (int)i);
        }

        TSmallVec<int> ints2 = ints;

        for (size_t i = 0; i < ints2.size(); ++i) {
            UNIT_ASSERT_EQUAL(ints2[i], (int)i);
        }

        TSmallVec<int> ints3;
        ints3 = ints2;

        for (size_t i = 0; i < ints3.size(); ++i) {
            UNIT_ASSERT_EQUAL(ints3[i], (int)i);
        }
    }

    Y_UNIT_TEST(TestCappedSize) {
        TStackVec<int, 8, false> ints;
        ints.push_back(1);
        ints.push_back(2);

        auto intsCopy = ints;
        UNIT_ASSERT_VALUES_EQUAL(intsCopy.capacity(), 8);

        for (int i = 2; i != 8; ++i) {
            intsCopy.push_back(i);
        }
        // Just verify that the program did not crash.
    }

    Y_UNIT_TEST(TestCappedSizeWithNotCopyAssignable) {
        TStackVec<TNotCopyAssignable, 8, false> values;
        values.push_back({1});
        values.push_back({2});

        auto valuesCopy = values;
        UNIT_ASSERT_VALUES_EQUAL(valuesCopy.capacity(), 8);

        for (int i = 2; i != 8; ++i) {
            valuesCopy.push_back({i});
        }
        // Just verify that the program did not crash.
    }
}
